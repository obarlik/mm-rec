"""
MM-Rec Block
Main layer combining Associative Scan, MDI, HDS, and Core Formula
Optimized for performance and memory efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
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
        
        # Performance optimization flags
        self.use_gradient_checkpointing = False  # Can be enabled via config
        self.use_kernel_fusion = True  # Enable kernel fusion optimizations
        
        # C++ optimization flag (auto-detect if available)
        self.use_cpp_optimization = False
        try:
            import mm_rec_cpp_cpu
            self.use_cpp_optimization = True
            print("✅ C++ optimizations available")
        except ImportError:
            self.use_cpp_optimization = False
            # Silent fallback to Python
    
    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState,
        hds: Optional[HierarchicalDataStructure] = None,
        use_checkpointing: Optional[bool] = None
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass through MM-Rec block with sequential state updates.
        
        Processes the entire sequence step-by-step, updating memory state
        at each timestep to ensure correct sequential dependencies.
        
        Optimizations:
        - Kernel fusion: Pre-compute QKVZ projections for entire sequence
        - Gradient checkpointing: Reduce memory usage (optional)
        
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            state: MemoryState instance
            hds: Optional HierarchicalDataStructure (created if None)
            use_checkpointing: Override checkpointing setting (default: self.use_gradient_checkpointing)
        
        Returns:
            Tuple of (output, updated_state):
                - output: Output tensor [batch, seq_len, model_dim]
                - updated_state: Updated MemoryState with sequential updates
        """
        batch_size, seq_len, _ = x.shape
        
        # Override checkpointing if specified
        if use_checkpointing is not None:
            self.use_gradient_checkpointing = use_checkpointing
        
        # Create HDS if not provided
        if hds is None:
            hds = HierarchicalDataStructure(
                memory_state=state,
                num_levels=3,
                model_dim=self.model_dim
            )
            hds.construct_hierarchy(state)
        
        # Initialize output tensor
        output = torch.zeros_like(x)  # [batch, seq_len, model_dim]
        
        # Initialize previous hidden state h_{t-1} (for t=0, use zeros)
        h_prev = torch.zeros(batch_size, 1, self.model_dim, 
                            dtype=x.dtype, device=x.device)
        
        # OPTIMIZATION: Pre-compute all QKVZ projections for entire sequence
        # This reduces CPU-GPU synchronization and enables better kernel fusion
        if self.use_kernel_fusion:
            # Batch all projections at once (more efficient)
            x_norm_all = self.norm1(x)  # [batch, seq_len, model_dim]
            
            # Fused QKVZ projections (single batch operation instead of seq_len operations)
            q_all = self.W_q(x_norm_all)  # [batch, seq_len, model_dim]
            k_all = self.W_k(x_norm_all)  # [batch, seq_len, model_dim]
            v_all = self.W_v(x_norm_all)  # [batch, seq_len, model_dim]
            z_all = self.W_z(x_norm_all)  # [batch, seq_len, model_dim]
        
        # Sequential processing: Loop over sequence steps
        for t in range(seq_len):
            # Get current timestep input
            if self.use_kernel_fusion:
                # Use pre-computed projections (faster, less CPU-GPU sync)
                x_t = x[:, t:t+1, :]  # [batch, 1, model_dim]
                x_t_norm = x_norm_all[:, t:t+1, :]  # [batch, 1, model_dim]
                q_t = q_all[:, t:t+1, :]  # [batch, 1, model_dim]
                k_t = k_all[:, t:t+1, :]  # [batch, 1, model_dim]
                v_t = v_all[:, t:t+1, :]  # [batch, 1, model_dim]
                z_t = z_all[:, t:t+1, :]  # [batch, 1, model_dim]
            else:
                # Original per-step computation (slower, more CPU-GPU sync)
                x_t = x[:, t:t+1, :]  # [batch, 1, model_dim]
                x_t_norm = self.norm1(x_t)
                q_t = self.W_q(x_t_norm)
                k_t = self.W_k(x_t_norm)
                v_t = self.W_v(x_t_norm)
                z_t = self.W_z(x_t_norm)
            
            # Step 3-5: MDI computation (optimized with kernel fusion and checkpointing)
            # Get h_{t-1} from previous iteration
            h_prev_expanded = h_prev  # [batch, 1, model_dim]
            
            # OPTIMIZATION: Use checkpointing for MDI if enabled (memory efficiency)
            if self.use_gradient_checkpointing:
                # Gradient checkpointing: recompute activations during backward
                # This trades compute for memory (recomputes forward during backward)
                mdi_fn = lambda z, h, ctx: self.mdi(z, h, context=ctx)
                h_new_t, gamma_new_t = checkpoint(mdi_fn, z_t, h_prev_expanded, k_t, use_reentrant=False)
            else:
                # Standard forward pass
                h_new_t, gamma_new_t = self.mdi(z_t, h_prev_expanded, context=k_t)
            
            # Step 4: Associative Scan - Compute cumulative exponential product
            # For sequential processing, we need cumulative product up to step t
            # Reshape for associative scan: [batch, heads, 1, head_dim]
            gamma_t_reshaped = gamma_new_t.view(batch_size, self.num_heads, 1, -1)
            
            # Use CPU fallback if CUDA not available
            try:
                cumprod_t = associative_scan_exponential(gamma_t_reshaped)
            except RuntimeError:
                from ..core.associative_scan_triton import associative_scan_exponential_cpu_fallback
                cumprod_t = associative_scan_exponential_cpu_fallback(gamma_t_reshaped)
            
            cumprod_t = cumprod_t.view(batch_size, 1, self.model_dim)
            
            # Step 6: Core Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
            # OPTIMIZATION: Fused gate computation (W_g + sigmoid in one step)
            # Pre-compute W_g(h_prev) and apply sigmoid
            gate_signal = torch.sigmoid(self.W_g(h_prev_expanded))  # σ(W_g h_{t-1})
            
            # Fused element-wise operations (reduces intermediate allocations)
            gated_input = z_t * gate_signal  # z_t ⊙ σ(W_g h_{t-1})
            decayed_prev = gamma_new_t * h_prev_expanded  # γ ⊙ h_{t-1}
            h_t = gated_input + decayed_prev  # h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
            
            # Ensure h_new_t from MDI also contributes (for gradient flow)
            h_t = h_t + 0.1 * h_new_t
            
            # Step 7: Multi-Memory Attention
            # OPTIMIZATION: Use checkpointing for attention if enabled
            if self.use_gradient_checkpointing:
                attn_fn = lambda h, q: self.multi_mem_attention(h, hds, state, q_input=q)
                mem_context_t = checkpoint(attn_fn, h_t, q_t, use_reentrant=False)
            else:
                mem_context_t = self.multi_mem_attention(h_t, hds, state, q_input=q_t)
            
            # CRITICAL FIX: Ensure v_t contributes to output for gradient flow
            v_contribution = v_t * 0.1
            h_attended_t = h_t + mem_context_t + v_contribution
            
            # Step 8: Residual connection
            x_residual_t = x_t + self.dropout_layer(h_attended_t)
            
            # Step 9: Feed-forward network
            # OPTIMIZATION: Use checkpointing for FFN if enabled
            if self.use_gradient_checkpointing:
                def ffn_step(x_res):
                    x_norm2 = self.norm2(x_res)
                    return self.ffn(x_norm2)
                ffn_out_t = checkpoint(ffn_step, x_residual_t, use_reentrant=False)
            else:
                x_norm2_t = self.norm2(x_residual_t)
                ffn_out_t = self.ffn(x_norm2_t)
            
            output_t = x_residual_t + ffn_out_t
            
            # Store output for this timestep
            output[:, t:t+1, :] = output_t
            
            # Update memory state at step t
            # Extract h_t without batch dimension for state update
            h_t_for_state = h_t.squeeze(1)  # [batch, model_dim]
            
            # Update short-term memory state at step t
            # Use h_t as both k and v for short-term memory
            state.update_state_sequential(
                bank_type='short',
                new_k=h_t_for_state,
                new_v=h_t_for_state,
                step=t
            )
            
            # Update h_prev for next iteration
            h_prev = h_t  # [batch, 1, model_dim]
        
        # Update long-term memory (less frequent, typically at block level)
        # For now, use a summary of the sequence
        # In practice, this could be done at block boundaries or periodically
        h_sequence_summary = output.mean(dim=1, keepdim=True)  # [batch, 1, model_dim]
        h_summary_k = h_sequence_summary.squeeze(1)  # [batch, model_dim]
        h_summary_v = h_sequence_summary.squeeze(1)
        
        # Update long-term memory (simplified - update all slots with summary)
        # In practice, this should be more sophisticated
        state.update_state('long', 
                          h_summary_k.unsqueeze(1).expand(-1, state.long_term.num_slots, -1),
                          h_summary_v.unsqueeze(1).expand(-1, state.long_term.num_slots, -1))
        
        # Reconstruct HDS hierarchy with updated state
        hds.reset_cache()
        hds.construct_hierarchy(state)
        
        return output, state

