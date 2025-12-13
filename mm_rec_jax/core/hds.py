import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple

from .memory_state import MemoryState, MemoryBank

class HDS:
    """
    Hierarchical Data Structure (JAX/Functional version).
    
    Provides O(M) memory access via multi-level pooling.
    Since JAX is functional, this class acts more like a namespace for 
    pure functions rather than a stateful object.
    """
    
    @staticmethod
    def construct_hierarchy(
        state: MemoryState, 
        num_levels: int = 3,
        level_ratios: Tuple[int, ...] = (4, 4) # Ratio between levels (0->1, 1->2)
    ) -> Dict[int, MemoryBank]:
        """
        Constructs hierarchy levels from long-term memory.
        
        Args:
            state: Current MemoryState
            num_levels: Total levels including base (0)
            level_ratios: Pooling stride for each subsequent level
            
        Returns:
            Dict mapping level_idx -> MemoryBank (containing pooled k, v)
        """
        # Level 0 is the raw long-term memory
        hierarchy = {0: state.long_term}
        
        current_k = state.long_term.k
        current_v = state.long_term.v
        
        # We assume [Batch, Slots, Dim] or [Slots, Dim]
        # MemoryBank initialization usually gives [Slots, Dim]
        # If we are inside vmap, it might have batch dim, but flax modules handle that.
        # But here we are using manual ops.
        
        # Ensure 3D for pooling: [Batch, Slots, Dim]
        # If 2D [Slots, Dim], add fake batch -> [1, Slots, Dim]
        added_batch = False
        if current_k.ndim == 2:
            current_k = current_k[None, ...]
            current_v = current_v[None, ...]
            added_batch = True
            
        for i in range(num_levels - 1):
            ratio = level_ratios[i] if i < len(level_ratios) else 4
            
            # Average Pooling
            # Input: [Batch, Length, Features]
            # Window: [1, ratio, 1]
            # Stride: [1, ratio, 1]
            
            current_k = nn.avg_pool(current_k, window_shape=(ratio,), strides=(ratio,), padding='VALID')
            current_v = nn.avg_pool(current_v, window_shape=(ratio,), strides=(ratio,), padding='VALID')
            
            # Store in hierarchy
            # We create a transient MemoryBank for the level
            # If we added batch, remove it for storage if needed, or keep it consistent.
            # Let's keep consistent: if input was 2D, output 2D.
            
            save_k = current_k[0] if added_batch else current_k
            save_v = current_v[0] if added_batch else current_v
            
            hierarchy[i + 1] = MemoryBank(k=save_k, v=save_v)
            
        return hierarchy

    @staticmethod
    def query_memory(
        hierarchy: Dict[int, MemoryBank],
        query: jnp.ndarray,
        level: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get Keys and Values for a specific hierarchy level.
        
        Args:
            hierarchy: Output from construct_hierarchy
            query: Query vector (used for shape broadcasting if needed)
            level: Level index to query
            
        Returns:
            (k, v) tensors for attention
        """
        bank = hierarchy[level]
        return bank.k, bank.v
