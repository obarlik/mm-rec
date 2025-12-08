"""
MM-Rec Multi-Memory Attention
O(M) complexity attention mechanism for long-term memory queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..core.hds import HierarchicalDataStructure
from ..core.memory_state import MemoryState


class MultiMemoryAttention(nn.Module):
    """
    Multi-Memory Attention mechanism for MM-Rec.
    
    Provides O(M) complexity attention instead of O(N²) by querying
    hierarchical long-term memory instead of full sequence.
    
    Where:
    - N: Sequence length (32K+)
    - M: Long-term memory size (typically 1024, M << N)
    
    Args:
        model_dim: Model dimension (hidden_dim)
        num_heads: Number of attention heads
        head_dim: Dimension per head (model_dim // num_heads)
    """
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else model_dim // num_heads
        
        assert model_dim == num_heads * self.head_dim, \
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        
        # Query projection (for h_t)
        self.W_q = nn.Linear(model_dim, model_dim)
        
        # Key and Value projections are not needed here because
        # we use Keys and Values directly from HDS memory
        
        # Output projection
        self.W_o = nn.Linear(model_dim, model_dim)
        
        # Scale factor for attention scores
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(
        self,
        query: torch.Tensor,
        hds: HierarchicalDataStructure,
        state: Optional[MemoryState] = None,
        q_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: Query hierarchical memory with O(M) complexity.
        
        Args:
            query: Query tensor (h_t) [batch, seq_len, model_dim]
            hds: HierarchicalDataStructure instance
            state: Optional MemoryState (for future use)
            q_input: Optional pre-computed query projection (to ensure W_q gradients)
        
        Returns:
            context: Contextualized output [batch, seq_len, model_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project query: Q = W_q · h_t
        # CRITICAL FIX: Always use attention's own W_q to ensure it receives gradients
        # If q_input is provided, combine it with attention's W_q output
        # This ensures both W_q (block) and W_q (attention) receive gradients
        q_attention = self.W_q(query)  # [batch, seq_len, model_dim] - Always compute for gradient flow
        if q_input is not None:
            # Combine block's q_input with attention's q_attention
            # This ensures both receive gradients
            q = q_attention + 0.5 * q_input  # Weighted combination
        else:
            q = q_attention  # Use attention's W_q output
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Query hierarchical memory at top level (smallest, O(M) complexity)
        k_mem, v_mem = hds.query_memory(query, level=-1)
        # k_mem: [batch, num_slots_M, model_dim]
        # v_mem: [batch, num_slots_M, model_dim]
        
        # Ensure dtype matches query
        k_mem = k_mem.to(query.dtype)
        v_mem = v_mem.to(query.dtype)
        
        # Reshape memory Keys and Values for multi-head
        k_mem = k_mem.view(batch_size, -1, self.num_heads, self.head_dim)
        k_mem = k_mem.transpose(1, 2)  # [batch, num_heads, num_slots_M, head_dim]
        
        v_mem = v_mem.view(batch_size, -1, self.num_heads, self.head_dim)
        v_mem = v_mem.transpose(1, 2)  # [batch, num_heads, num_slots_M, head_dim]
        
        # Compute attention scores: scores = Q · K_mem^T / sqrt(d_k)
        # CRITICAL: This is O(N*M) not O(N²) because M << N (M=1024, N=100K)
        # However, for very long sequences (100K+), this can still be large.
        # Memory: batch * num_heads * seq_len * num_slots_M * 4 bytes (FP32)
        # For 100K seq_len: 1 * 8 * 100000 * 1024 * 4 = ~3.2 GB (acceptable but large)
        scores = torch.matmul(q, k_mem.transpose(-2, -1)) * self.scale
        # scores: [batch, num_heads, seq_len, num_slots_M]
        
        # MEMORY CHECK: Warn if attention scores matrix is too large
        scores_size_mb = scores.numel() * scores.element_size() / (1024 ** 2)
        if scores_size_mb > 1000:  # > 1 GB
            import warnings
            warnings.warn(
                f"⚠️ Large attention scores matrix: {scores_size_mb:.2f} MB\n"
                f"   Shape: {scores.shape}, Memory: O(N*M) where N={seq_len}, M={k_mem.shape[-2]}\n"
                f"   Consider using chunking for sequences > 32K.",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [batch, num_heads, seq_len, num_slots_M]
        
        # Compute context: context = softmax(scores) · V_mem
        context = torch.matmul(attn_weights, v_mem)
        # context: [batch, num_heads, seq_len, head_dim]
        
        # Reshape back: [batch, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.model_dim)
        
        # Output projection
        output = self.W_o(context)  # [batch, seq_len, model_dim]
        
        return output

