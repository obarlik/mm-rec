import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, Optional

# Define MemoryState as a PyTreeNode so it can be passed through JIT/Scan boundaries
@struct.dataclass
@struct.dataclass
class MemoryBank(struct.PyTreeNode):
    k: jnp.ndarray
    v: jnp.ndarray
    age: Optional[jnp.ndarray] = None
    usage: Optional[jnp.ndarray] = None
    idx: Optional[jnp.ndarray] = None # Ring buffer pointer [Batch] or Scalar

@struct.dataclass
class MemoryState(struct.PyTreeNode):
    short_term: MemoryBank
    long_term: MemoryBank

    @classmethod
    def create(cls, 
               short_dim: int, 
               short_len: int, 
               long_dim: int, 
               long_len: int, 
               dtype=jnp.float32):
        """Initialize empty memory state."""
        return cls(
            short_term=MemoryBank(
                k=jnp.zeros((short_len, short_dim), dtype=dtype),
                v=jnp.zeros((short_len, short_dim), dtype=dtype),
                age=jnp.zeros((short_len,), dtype=jnp.int32),
                idx=jnp.array(0, dtype=jnp.int32)
            ),
            long_term=MemoryBank(
                k=jnp.zeros((long_len, long_dim), dtype=dtype),
                v=jnp.zeros((long_len, long_dim), dtype=dtype),
                usage=jnp.zeros((long_len,), dtype=jnp.float32),
                idx=jnp.array(0, dtype=jnp.int32)
            )
        )

    def update_short(self, k_new: jnp.ndarray, v_new: jnp.ndarray):
        """
        Functional update for short-term memory (Ring Buffer).
        Uses O(1) dynamic_update_slice.
        """
        # Chunk Update: [Batch, Seq, Dim]
        # Single Update: [Batch, Dim]
        
        is_chunk = k_new.ndim == 3
        
        # Handle Unbatched / Init case
        current_k = self.short_term.k
        current_v = self.short_term.v
        current_age = self.short_term.age
        current_idx = self.short_term.idx
        
        # Normalize inputs to [Batch, Seq, Dim]
        if not is_chunk:
             # Make it [Batch, 1, Dim]
             k_new = k_new[:, None, :]
             v_new = v_new[:, None, :]
             
        batch_size, seq_len, dim = k_new.shape
        num_slots = current_k.shape[1] if current_k.ndim == 3 else current_k.shape[0]

        # Broadcast state if unbatched
        if current_k.ndim == 2:
            current_k = jnp.broadcast_to(current_k[None, ...], (batch_size, num_slots, dim))
            current_v = jnp.broadcast_to(current_v[None, ...], (batch_size, num_slots, dim))
            current_age = jnp.broadcast_to(current_age[None, ...], (batch_size, num_slots))
            current_idx = jnp.broadcast_to(current_idx[None, ...], (batch_size,))
            
        # Ring Buffer Update Logic
        # We need to write 'k_new' into 'current_k' starting at 'current_idx'.
        # If seq_len > num_slots, we only take the last num_slots.
        
        if seq_len > num_slots:
            k_new = k_new[:, -num_slots:, :]
            v_new = v_new[:, -num_slots:, :]
            seq_len = num_slots
            
        # Indices for scatter
        # We use dynamic_update_slice.
        # But JAX doesn't support "wrapping" dynamic_update_slice automatically.
        # So we might need two updates if it crosses the boundary.
        
        # Calculate indices
        # Vectorized over batch? dynamic_update_slice usually takes scalar start_idx for dim 0, or ...
        # Actually, we want to update each batch item at potentially different indices?
        # No, if 'idx' is synced across batch (it should be if we start sync), we can use one idx.
        # But let's assume batched idx for robustness.
        
        # BUT vmap(dynamic_update_slice) is efficient.
        
        def update_single_batch(k, v, age, idx, new_k, new_v):
            # k: [Slots, Dim], idx: scalar
            # new_k: [Seq, Dim]
            
            # 1. Update Age (Global increment)
            # age = age + seq_len 
            # (Simple approximation: age is just time counter)
            age = age + seq_len
            
            # Write new data
            # Check if wrap around
            remaining = num_slots - idx
            
            def write_wrapped(k, v, age, idx, new_k, new_v):
                 # Case 1: No Wrap
                 # Just one update
                 k = jax.lax.dynamic_update_slice(k, new_k, (idx, 0))
                 v = jax.lax.dynamic_update_slice(v, new_v, (idx, 0))
                 
                 # Update age for new items to 0
                 new_age_vals = jnp.arange(seq_len)[::-1] # 0 is newest
                 age = jax.lax.dynamic_update_slice(age, new_age_vals, (idx,))
                 
                 return k, v, age
                 
            def write_split(k, v, age, idx, new_k, new_v):
                 # Case 2: Wrap Around
                 # Part 1: idx to end
                 part1_len = num_slots - idx
                 part1_k = new_k[:part1_len]
                 part1_v = new_v[:part1_len]
                 
                 k = jax.lax.dynamic_update_slice(k, part1_k, (idx, 0))
                 v = jax.lax.dynamic_update_slice(v, part1_v, (idx, 0))
                 
                 part1_age = jnp.arange(seq_len)[::-1][:part1_len]
                 age = jax.lax.dynamic_update_slice(age, part1_age, (idx,))
                 
                 # Part 2: 0 to remaining
                 part2_len = seq_len - part1_len
                 part2_k = new_k[part1_len:]
                 part2_v = new_v[part1_len:]
                 
                 k = jax.lax.dynamic_update_slice(k, part2_k, (0, 0))
                 v = jax.lax.dynamic_update_slice(v, part2_v, (0, 0))
                 
                 part2_age = jnp.arange(seq_len)[::-1][part1_len:]
                 age = jax.lax.dynamic_update_slice(age, part2_age, (0,))
                 
                 return k, v, age

            # JAX Condition
            # If seq_len <= remaining, use write_wrapped
            # Else write_split
            k, v, age = jax.lax.cond(
                seq_len <= remaining,
                write_wrapped,
                write_split,
                k, v, age, idx, new_k, new_v
            )
            
            new_idx = (idx + seq_len) % num_slots
            return k, v, age, new_idx

        # VMAP over batch
        # k: [B, S, D], idx: [B]
        new_k, new_v, new_age, new_idx = jax.vmap(update_single_batch)(
            current_k, current_v, current_age, current_idx, k_new, v_new
        )
        
        new_bank = self.short_term.replace(k=new_k, v=new_v, age=new_age, idx=new_idx)
        return self.replace(short_term=new_bank)

    def update_long(self, k_new, v_new, mask=None):
        """Functional update for long-term memory (LIFO/Usage based)."""
        # Placeholder for simple update (replace least used or oldest)
        # For now, let's assume no-op or simple slot overwrite for migration start
        return self
