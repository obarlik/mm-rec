import jax
import jax.numpy as jnp
from flax import struct, serialization
from typing import Dict, Optional, Tuple, Any
import os

# Define MemoryState as a PyTreeNode so it can be passed through JIT/Scan boundaries
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
            
            # Scatter Update via .at[].set()
            # This handles wrapping automatically via modulo arithmetic on indices
            
            # 1. Update Age (Global increment)
            age = age + seq_len
            
            # 2. Calculate indices for new data
            # [0, 1, ..., seq_len-1] + idx % num_slots
            # We wrap indices so they fit in [0, num_slots-1]
            indices = (idx + jnp.arange(seq_len)) % num_slots
            
            # 3. Perform Update
            # JAX handles parallel/conflicting indices (last write wins usually, but here unique)
            k = k.at[indices].set(new_k)
            v = v.at[indices].set(new_v)
            
            # 4. Update Age for new items
            new_age_vals = jnp.arange(seq_len)[::-1] # 0 is newest
            age = age.at[indices].set(new_age_vals)
            
            # 5. Advance Index
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

    # ========================================================================
    # Persistence & Serialization (Phase 8: Decoupled Storage)
    # ========================================================================
    
    def to_bytes(self) -> bytes:
        """Serialize memory state to msgpack bytes."""
        return serialization.to_bytes(self)
        
    @classmethod
    def from_bytes(cls, data: bytes, template: 'MemoryState') -> 'MemoryState':
        """
        Deserialize memory state from msgpack bytes.
        
        Args:
            data: Binary msgpack data.
            template: A dummy MemoryState instance with the same structure/shapes.
                      JAX/Flax needs this to know the structure.
        """
        return serialization.from_bytes(template, data)
        
    def save(self, path: str):
        """Save to disk (atomic write)."""
        temp_path = f"{path}.tmp"
        with open(temp_path, "wb") as f:
            f.write(self.to_bytes())
        os.rename(temp_path, path)
        
    @classmethod
    def load(cls, path: str, template: 'MemoryState') -> 'MemoryState':
        """Load from disk."""
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return cls.from_bytes(f.read(), template)
