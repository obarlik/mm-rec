"""
MM-Rec Hierarchical Data Structure (HDS)
Implements O(M) memory access with multi-level hierarchy
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .memory_state import MemoryState


class HierarchicalDataStructure(nn.Module):
    """
    Hierarchical Data Structure for efficient memory access.
    
    Provides O(M) access cost instead of O(N) where:
    - N: Sequence length (32K+)
    - M: Long-term memory size (typically 1024, M << N)
    
    Hierarchy levels:
    - Level 0: Token-level (long-term memory M)
    - Level 1: Block-level (summaries of blocks)
    - Level 2: Global-level (summaries of sequences)
    - Level 3: Long-term M (persistent memory)
    
    Args:
        memory_state: MemoryState instance containing memory banks
        num_levels: Number of hierarchy levels (default: 3)
        level_dims: List of memory slot counts for each level
        model_dim: Model dimension
    """
    
    def __init__(
        self,
        memory_state: MemoryState,
        num_levels: int = 3,
        level_dims: Optional[List[int]] = None,
        model_dim: int = 4096
    ):
        super().__init__()
        self.memory_state = memory_state
        self.num_levels = num_levels
        self.model_dim = model_dim
        
        # Default level dimensions if not provided
        # Level 0: Full long-term memory (M slots)
        # Level 1: Block summaries (M // 4 slots)
        # Level 2: Global summaries (M // 16 slots)
        if level_dims is None:
            # Assume M = 1024 for default
            M = memory_state.long_term.num_slots
            level_dims = [M, M // 4, M // 16]
        
        self.level_dims = level_dims[:num_levels]
        
        # Pooling layers for hierarchy construction
        # Each level pools from the previous level
        self.pooling_layers = nn.ModuleList()
        for i in range(num_levels - 1):
            # Average pooling to reduce dimensions
            pool = nn.AdaptiveAvgPool1d(self.level_dims[i + 1])
            self.pooling_layers.append(pool)
        
        # Cache for constructed hierarchy
        self.levels_cache = {}
        self._hierarchy_constructed = False
    
    def construct_hierarchy(self, state: Optional[MemoryState] = None) -> None:
        """
        Construct hierarchical memory structure from MemoryState.
        
        This creates multi-level summaries of the long-term memory,
        enabling efficient O(M) queries instead of O(N).
        
        Args:
            state: MemoryState to construct hierarchy from (if None, uses self.memory_state)
        """
        if state is None:
            state = self.memory_state
        
        # Get long-term memory Key and Value tensors
        k_long, v_long = state.get_state('long')
        
        # Level 0: Original long-term memory (M slots)
        # Shape: [num_slots, k_dim] or [batch, num_slots, k_dim]
        if k_long.dim() == 2:
            # [num_slots, k_dim] - add batch dimension
            k_level_0 = k_long.unsqueeze(0)  # [1, num_slots, k_dim]
            v_level_0 = v_long.unsqueeze(0)  # [1, num_slots, v_dim]
        else:
            k_level_0 = k_long  # [batch, num_slots, k_dim]
            v_level_0 = v_long  # [batch, num_slots, v_dim]
        
        self.levels_cache[0] = {
            'k': k_level_0,
            'v': v_level_0,
            'num_slots': self.level_dims[0]
        }
        
        # Build higher levels by pooling
        k_current = k_level_0
        v_current = v_level_0
        
        for level_idx in range(1, self.num_levels):
            # Pool from previous level
            # Transpose for pooling: [batch, dim, num_slots]
            k_pooled = self.pooling_layers[level_idx - 1](
                k_current.transpose(1, 2)  # [batch, k_dim, num_slots]
            ).transpose(1, 2)  # [batch, num_slots_new, k_dim]
            
            v_pooled = self.pooling_layers[level_idx - 1](
                v_current.transpose(1, 2)  # [batch, v_dim, num_slots]
            ).transpose(1, 2)  # [batch, num_slots_new, v_dim]
            
            self.levels_cache[level_idx] = {
                'k': k_pooled,
                'v': v_pooled,
                'num_slots': self.level_dims[level_idx]
            }
            
            k_current = k_pooled
            v_current = v_pooled
        
        self._hierarchy_constructed = True
    
    def query_memory(
        self,
        query: torch.Tensor,
        level: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query hierarchical memory at specified level.
        
        This is the O(M) access interface used by Multi-Memory Attention.
        The query is compared against Keys at the specified level,
        and corresponding Values are returned.
        
        Args:
            query: Query tensor (h_t) [batch, seq_len, model_dim] or [batch, model_dim]
            level: Hierarchy level to query (-1 = top level, smallest memory)
        
        Returns:
            Tuple of (k_level, v_level):
                - k_level: Key tensors at specified level [batch, num_slots_level, k_dim]
                - v_level: Value tensors at specified level [batch, num_slots_level, v_dim]
        """
        # Ensure hierarchy is constructed
        if not self._hierarchy_constructed:
            self.construct_hierarchy()
        
        # Determine level index
        if level < 0:
            level = self.num_levels - 1  # Top level (smallest memory)
        
        level = min(level, self.num_levels - 1)
        
        # Get Key and Value tensors for specified level
        level_data = self.levels_cache[level]
        k_level = level_data['k']  # [batch, num_slots, k_dim]
        v_level = level_data['v']  # [batch, num_slots, v_dim]
        
        # Handle query shape
        if query.dim() == 2:
            # [batch, model_dim] - add seq_len dimension
            query = query.unsqueeze(1)  # [batch, 1, model_dim]
        
        # Ensure batch dimension matches
        if k_level.shape[0] == 1 and query.shape[0] > 1:
            # Broadcast k_level and v_level to match query batch size
            k_level = k_level.expand(query.shape[0], -1, -1)
            v_level = v_level.expand(query.shape[0], -1, -1)
        
        return k_level, v_level
    
    def get_level_info(self, level: int) -> dict:
        """
        Get information about a specific hierarchy level.
        
        Args:
            level: Level index
        
        Returns:
            Dictionary with level information (num_slots, etc.)
        """
        if not self._hierarchy_constructed:
            self.construct_hierarchy()
        
        if level < 0:
            level = self.num_levels - 1
        
        level = min(level, self.num_levels - 1)
        
        return {
            'level': level,
            'num_slots': self.level_dims[level],
            'k_shape': self.levels_cache[level]['k'].shape,
            'v_shape': self.levels_cache[level]['v'].shape
        }
    
    def reset_cache(self):
        """Reset hierarchy cache (call when memory state changes)."""
        self.levels_cache = {}
        self._hierarchy_constructed = False

