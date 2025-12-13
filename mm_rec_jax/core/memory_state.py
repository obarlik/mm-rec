import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, Optional

# Define MemoryState as a PyTreeNode so it can be passed through JIT/Scan boundaries
@struct.dataclass
class MemoryBank(struct.PyTreeNode):
    k: jnp.ndarray
    v: jnp.ndarray
    age: Optional[jnp.ndarray] = None
    usage: Optional[jnp.ndarray] = None

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
                age=jnp.zeros((short_len,), dtype=jnp.int32)
            ),
            long_term=MemoryBank(
                k=jnp.zeros((long_len, long_dim), dtype=dtype),
                v=jnp.zeros((long_len, long_dim), dtype=dtype),
                usage=jnp.zeros((long_len,), dtype=jnp.float32)
            )
        )

    def update_short(self, k_new: jnp.ndarray, v_new: jnp.ndarray):
        """
        Functional update for short-term memory (FIFO).
        Handling both single item [Batch, Dim] and chunk [Batch, Seq, Dim].
        """
        # Detect if chunk or single item
        # k_new shape: [Batch, Dim] or [Batch, Seq, Dim]
        # self.short_term.k shape: [Batch, Slots, Dim]
        
        is_chunk = k_new.ndim == 3
        
        if is_chunk:
            # Chunk Update: [Batch, Seq, Dim]
            seq_len = k_new.shape[1]
            num_slots = self.short_term.k.shape[1]
            
            # Shift left by seq_len
            # [Batch, Slots-Seq, Dim]
            kept_k = self.short_term.k[:, seq_len:, :]
            kept_v = self.short_term.v[:, seq_len:, :]
            kept_age = self.short_term.age[:, seq_len:] # [Batch, Slots-Seq]
            
            # Append new
            new_k = jnp.concatenate([kept_k, k_new], axis=1)
            new_v = jnp.concatenate([kept_v, v_new], axis=1)
            
            # Update age
            # Increment existing
            new_kept_age = kept_age + seq_len
            # New items have age 0, 1, ..., seq_len-1 ? Or all 0?
            # Typically age is "time since arrival".
            # The last item in chunk is freshest (0). First item is oldest in chunk (seq_len-1).
            # Let's assign 0s for now for simplicity or proper range.
            # actually we can just use 0 for all new items if we treat chunk as one block, 
            # but element-wise age is better.
            zeros = jnp.zeros((k_new.shape[0], seq_len), dtype=jnp.int32)
            new_age = jnp.concatenate([new_kept_age, zeros], axis=1)
            
        else:
            # Single Item Update: [Batch, Dim]
            # Shift left by 1
            new_k = jnp.concatenate([self.short_term.k[:, 1:, :], k_new[:, None, :]], axis=1)
            new_v = jnp.concatenate([self.short_term.v[:, 1:, :], v_new[:, None, :]], axis=1)
            
            new_age = jnp.concatenate([self.short_term.age[:, 1:] + 1, jnp.zeros((k_new.shape[0], 1), dtype=jnp.int32)], axis=1)
        
        new_bank = self.short_term.replace(k=new_k, v=new_v, age=new_age)
        return self.replace(short_term=new_bank)

    def update_long(self, k_new, v_new, mask=None):
        """Functional update for long-term memory (LIFO/Usage based)."""
        # Placeholder for simple update (replace least used or oldest)
        # For now, let's assume no-op or simple slot overwrite for migration start
        return self
