
import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional

# Defines
short_len = 512
short_dim = 512
batch_size = 16

@struct.dataclass
class MemoryBank(struct.PyTreeNode):
    k: jnp.ndarray
    v: jnp.ndarray
    idx: Optional[jnp.ndarray] = None

@struct.dataclass
class MemoryState(struct.PyTreeNode):
    short_term: MemoryBank
    long_term: MemoryBank # Simplified

def create_state():
    return MemoryState(
        short_term=MemoryBank(
            k=jnp.zeros((short_len, short_dim)),
            v=jnp.zeros((short_len, short_dim)),
            idx=jnp.array(0, dtype=jnp.int32)
        ),
        long_term=MemoryBank(
            k=jnp.zeros((short_len, short_dim)),
            v=jnp.zeros((short_len, short_dim)),
            idx=jnp.array(0, dtype=jnp.int32)
        )
    )

def gather_roll(arr, idx):
    # arr: [S, D]
    num_slots = arr.shape[0]
    indices = (jnp.arange(num_slots) + idx) % num_slots
    return arr[indices]

def main():
    print("🚀 Reproducing Dimensions Issue...")
    base_state = create_state()
    print(f"Base State k shape: {base_state.short_term.k.shape}")
    print(f"Base State idx shape: {base_state.short_term.idx.shape}")

    # Broadcast
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), 
        base_state
    )
    print(f"Batched State k shape: {batched_state.short_term.k.shape}")
    print(f"Batched State idx shape: {batched_state.short_term.idx.shape}")
    
    current_k = batched_state.short_term.k
    current_idx = batched_state.short_term.idx # Should be (16,)
    
    # Simulate HDS.construct_hierarchy
    # if current_k.ndim == 3: # Batched
    print(f"Current K ndim: {current_k.ndim}")
    
    mapped_fn = jax.vmap(gather_roll)
    result_k = mapped_fn(current_k, current_idx)
    
    print(f"Result K shape: {result_k.shape}")
    
    if result_k.ndim == 5:
        print("❌ REPRODUCED Rank 5 Tensor!")
    else:
        print("✅ Result is not Rank 5. (But check if it's correct)")

if __name__ == "__main__":
    main()
