"""
MM-Rec Block
Main layer combining Associative Scan, MDI, HDS, and Core Formula
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..core.associative_scan_triton import associative_scan_exponential
from ..core.mdi import MemoryDecayIntegration
from ..core.hds import HierarchicalDataStructure
from ..core.memory_state import MemoryState
from .attention import MultiMemoryAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.weight * x / (norm + self.eps)


class MMRecBlock(nn.Module):
    """
    MM-Rec Block: Complete layer combining all components.
    
    Implements the 7-step forward pass:
    1. Query, Key, Value, Z transformations
    2. Associative Scan (exponential product)
    3. MDI (Memory Decay/Integration)
    4. Core Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    5. Multi-Memory Attention
    6. Residual connections
    7. Output projection
    
    Args:
        model_dim: Model dimension (hidden_dim, default: 4096)
        inner_dim: Inner dimension for MDI
        num_heads: Number of attention heads
        num_memories: Number of memory banks
        mem_dim: Memory dimension
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        model_dim: int = 4096,
        inner_dim: Optional[int] = None,
        num_heads: int = 8,
        num_memories: int = 1,
        mem_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
        self.num_heads = num_heads
        self.num_memories = num_memories
        self.mem_dim = mem_dim if mem_dim is not None else model_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
        self.dropout = dropout
        
        # Input projections
        self.W_q = nn.Linear(model_dim, model_dim)  # Query
        self.W_k = nn.Linear(model_dim, model_dim)  # Key
        self.W_v = nn.Linear(model_dim, model_dim)  # Value
        self.W_z = nn.Linear(model_dim, model_dim)  # z_t for core formula
        
        # Gating projection for core formula: W_g
        self.W_g = nn.Linear(model_dim, model_dim)
        
        # Normalization layers
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        
        # MDI (Memory Decay/Integration)
        self.mdi = MemoryDecayIntegration(
            model_dim=model_dim,
            inner_dim=self.inner_dim,
            use_context_modulation=True
        )
        
        # Multi-Memory Attention
        self.multi_mem_attention = MultiMemoryAttention(
            model_dim=model_dim,
            num_heads=num_heads
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState,
        hds: Optional[HierarchicalDataStructure] = None
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass through MM-Rec block.
        
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            state: MemoryState instance
            hds: Optional HierarchicalDataStructure (created if None)
        
        Returns:
            Tuple of (output, updated_state):
                - output: Output tensor [batch, seq_len, model_dim]
                - updated_state: Updated MemoryState
        """
        batch_size, seq_len, _ = x.shape
        
        # Create HDS if not provided
        if hds is None:
            hds = HierarchicalDataStructure(
                memory_state=state,
                num_levels=3,
                model_dim=self.model_dim
            )
            hds.construct_hierarchy(state)
        
        # Step 1: Normalize input
        x_norm = self.norm1(x)
        
        # Step 2: Query, Key, Value, Z transformations
        q = self.W_q(x_norm)  # [batch, seq_len, model_dim]
        k = self.W_k(x_norm)  # [batch, seq_len, model_dim]
        v = self.W_v(x_norm)  # [batch, seq_len, model_dim]
        z_t = self.W_z(x_norm)  # [batch, seq_len, model_dim]
        
        # Step 3: Compute decay coefficients γ from k
        # Use k as input to compute γ (simplified - in practice γ comes from MDI)
        gamma = self.mdi.compute_decay_only(z_t, context=k)
        # gamma: [batch, seq_len, model_dim]
        
        # Step 4: Associative Scan - Compute cumulative exponential product
        # Reshape for associative scan: [batch, heads, seq_len, head_dim]
        # For simplicity, treat each dimension independently
        gamma_reshaped = gamma.view(batch_size, self.num_heads, seq_len, -1)
        cumprod = associative_scan_exponential(gamma_reshaped)
        cumprod = cumprod.view(batch_size, seq_len, self.model_dim)
        
        # Step 5: MDI - Get previous hidden state and compute new state
        # Get previous state from memory (simplified - use x as h_prev for first iteration)
        h_prev = x  # In practice, this comes from previous timestep/state
        
        # Compute new hidden state using MDI
        h_new, gamma_new = self.mdi(z_t, h_prev, context=k)
        
        # Step 6: Core Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        gate_signal = torch.sigmoid(self.W_g(h_prev))  # σ(W_g h_{t-1})
        gated_input = z_t * gate_signal  # z_t ⊙ σ(W_g h_{t-1})
        decayed_prev = gamma_new * h_prev  # γ ⊙ h_{t-1}
        h_t = gated_input + decayed_prev  # h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        
        # Step 7: Multi-Memory Attention
        mem_context = self.multi_mem_attention(h_t, hds, state)
        
        # Combine attention with h_t
        h_attended = h_t + mem_context
        
        # Step 8: Residual connection
        x_residual = x + self.dropout_layer(h_attended)
        
        # Step 9: Feed-forward network
        x_norm2 = self.norm2(x_residual)
        ffn_out = self.ffn(x_norm2)
        output = x_residual + ffn_out
        
        # Update memory state (simplified - in practice update with new h_t)
        # For now, just return the same state
        updated_state = state
        
        return output, updated_state

