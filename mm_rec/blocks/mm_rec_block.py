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
try:
    from ..core.jax_connector import JaxScanner
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False



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
        dropout: float = 0.1,
        # HEM Parameters
        use_hem: bool = False,           # Enable HEM (Fused Kernel) mechanism
        pe_dim: Optional[int] = None,    # Positional encoding dimension (default: model_dim)
        # DPG Parameters
        use_dpg: bool = False,           # Enable DPG (Dynamic Projection Gating) mechanism
        dpg_rank: int = 128,             # Low-rank projection dimension (D -> 128 -> D)
        # Sparse FFN Parameters
        use_sparse: bool = False,        # Enable Sparse LSH FFN
        sparse_chunk_size: int = 128,    # Chunk size for sparse routing
        num_experts: int = 64            # Number of experts
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
        self.num_heads = num_heads
        self.num_memories = num_memories
        self.mem_dim = mem_dim if mem_dim is not None else model_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
        self.dropout = dropout
        
        # HEM Configuration
        self.use_hem = use_hem
        self.pe_dim = pe_dim if pe_dim is not None else model_dim
        
        # DPG Configuration
        self.use_dpg = use_dpg
        self.dpg_rank = dpg_rank  # Low-rank dimension: 128
        
        # ========================================================================
        # HEM: Fused Kernel - Single Large Weight Matrix
        # ========================================================================
        # HEM mekanizması, altı projeksiyonu (QKVZ + PE) tek bir büyük ağırlık
        # matrisi olarak birleştirir:
        # 
        # W_fused = [W_Q; W_K; W_V; W_Z; W_P; W_E]
        # 
        # Fused Matrix Shape:
        # - Input:  [batch, seq_len, model_dim]
        # - Weight: [model_dim, 4*model_dim + pe_dim + model_dim]
        #          = [model_dim, 5*model_dim + pe_dim]
        # - Output: [batch, seq_len, 5*model_dim + pe_dim]
        # ========================================================================
        
        if self.use_hem:
            # Calculate fused output dimension
            # Q + K + V + Z + P_down + E_up
            # = model_dim + model_dim + model_dim + model_dim + pe_dim + model_dim
            # = 5*model_dim + pe_dim
            self.fused_out_dim = 4 * model_dim + self.pe_dim + model_dim  # QKVZ + P + E
            
            # CRITICAL: Single fused linear layer
            # This replaces 6 separate Linear layers with 1 large Linear layer
            self.W_fused = nn.Linear(
                in_features=model_dim,
                out_features=self.fused_out_dim,
                bias=True  # Bias term for each projection
            )
            
            # Store projection dimensions for splitting
            self.proj_dims = {
                'Q': model_dim,
                'K': model_dim,
                'V': model_dim,
                'Z': model_dim,
                'P': self.pe_dim,  # Positional encoding down-projection
                'E': model_dim     # Positional encoding up-projection
            }
            
            # Calculate split indices for output tensor
            # Output: [Q, K, V, Z, P, E]
            self.split_indices = [
                self.proj_dims['Q'],                                    # Q end
                self.proj_dims['Q'] + self.proj_dims['K'],              # K end
                self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'],  # V end
                self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'] + self.proj_dims['Z'],  # Z end
                self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'] + self.proj_dims['Z'] + self.proj_dims['P'],  # P end
                self.fused_out_dim  # E end (total)
            ]
            
            # Initialize fused weight matrix
            # CRITICAL: Proper initialization for each sub-matrix
            self._init_fused_weights()
        else:
            # Fallback: Separate projections (original approach)
            self.W_fused = None
            self.proj_dims = None
            self.split_indices = None
            
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
        
        # ========================================================================
        # DPG: Dynamic Projection Gating - Low-Rank Projeksiyonlar
        # ========================================================================
        # DPG mekanizması için low-rank projeksiyonlar:
        # γ_t = σ(W_γ,up · ReLU(W_γ,down · z_t))
        # 
        # Boyutlar:
        # - W_γ,down: [model_dim, dpg_rank] = [4096, 128] (down-projection)
        # - W_γ,up:   [dpg_rank, model_dim] = [128, 4096] (up-projection)
        # 
        # Bu low-rank yapı sayesinde:
        # - Parametre sayısı: 4096×128 + 128×4096 = 1,048,576 (full: 4096×4096 = 16,777,216)
        # - 16x parametre tasarrufu
        # - Daha hızlı hesaplama
        # ========================================================================
        
        if self.use_dpg:
            # W_γ,down: Down-projection (D -> 128)
            # Input: z_t [batch, seq_len, model_dim]
            # Output: [batch, seq_len, dpg_rank]
            self.W_gamma_down = nn.Linear(
                in_features=model_dim,
                out_features=dpg_rank,
                bias=True  # Bias term for flexibility
            )
            
            # W_γ,up: Up-projection (128 -> D)
            # Input: [batch, seq_len, dpg_rank]
            # Output: [batch, seq_len, model_dim]
            self.W_gamma_up = nn.Linear(
                in_features=dpg_rank,
                out_features=model_dim,
                bias=True  # Bias term for flexibility
            )
            
            # Activation: ReLU between down and up projections
            # This adds non-linearity and ensures non-negative intermediate values
            self.dpg_activation = nn.ReLU()
            
            # Final activation: Sigmoid to ensure γ_t ∈ [0, 1]
            # Applied after up-projection
            self.dpg_sigmoid = nn.Sigmoid()
        else:
            # Fallback: Use full-rank projection (original MDI approach)
            # This is kept for backward compatibility
            self.W_gamma_down = None
            self.W_gamma_up = None
            self.dpg_activation = None
            self.dpg_sigmoid = None
        
        # MDI (Memory Decay/Integration)
        # Note: use_uboo will be passed from model level
        self.mdi = MemoryDecayIntegration(
            model_dim=model_dim,
            inner_dim=self.inner_dim,
            use_context_modulation=True,
            use_uboo=False  # Will be set from model level if needed
        )
        
        # Multi-Memory Attention
        self.multi_mem_attention = MultiMemoryAttention(
            model_dim=model_dim,
            num_heads=num_heads
        )
        
        # Feed-forward network
        # Feed-forward network (Dense or Sparse)
        if use_sparse:
            # Lazy import to avoid circular dependency if not needed
            from .sparse_mm_rec_block import SparseFFN
            self.ffn = SparseFFN(
                model_dim=model_dim,
                chunk_size=sparse_chunk_size,
                num_experts=num_experts,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
        else:
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
        
        # C++ optimization flag (REQUIRED on CPU)
        self.use_cpp_optimization = False
        try:
            # Preload PyTorch libraries to fix libc10.so issues
            import ctypes
            import os
            import torch
            
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib):
                libc10_path = os.path.join(torch_lib, 'libc10.so')
                if os.path.exists(libc10_path):
                    try:
                        ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
                os.environ['LD_LIBRARY_PATH'] = torch_lib
            
            import mm_rec_cpp_cpu
            self.use_cpp_optimization = True
            # Only print once per process (not per block instance)
            if not hasattr(MMRecBlock, '_cpp_optimization_printed'):
                print("✅ C++ optimizations available")
                MMRecBlock._cpp_optimization_printed = True
        except ImportError as e:
            # On CPU, C++ extension is REQUIRED
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"❌ CRITICAL: C++ extension 'mm_rec_cpp_cpu' is REQUIRED for CPU mode!\n"
                    f"   Error: {e}\n"
                    f"   Solution: cd mm_rec/cpp && python setup.py build_ext --inplace\n"
                    f"   Fallback to Python is DISABLED for performance."
                ) from e
            # On GPU, C++ is optional (Triton preferred)
            self.use_cpp_optimization = False
    
    def _init_fused_weights(self):
        """
        Initialize fused weight matrix by properly initializing each sub-matrix.
        
        CRITICAL: Each projection (Q, K, V, Z, P, E) should be initialized
        independently to maintain proper weight initialization.
        """
        with torch.no_grad():
            # Get fused weight and bias
            fused_weight = self.W_fused.weight.data  # [fused_out_dim, model_dim]
            fused_bias = self.W_fused.bias.data      # [fused_out_dim]
            
            # Initialize each sub-matrix with proper initialization
            # Standard initialization: Xavier/Glorot uniform for linear layers
            
            # Q projection: [model_dim, model_dim]
            start_idx = 0
            end_idx = self.proj_dims['Q']
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # K projection: [model_dim, model_dim]
            start_idx = end_idx
            end_idx += self.proj_dims['K']
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # V projection: [model_dim, model_dim]
            start_idx = end_idx
            end_idx += self.proj_dims['V']
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # Z projection: [model_dim, model_dim]
            start_idx = end_idx
            end_idx += self.proj_dims['Z']
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # P projection (down): [pe_dim, model_dim]
            start_idx = end_idx
            end_idx += self.proj_dims['P']
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # E projection (up): [model_dim, pe_dim]
            start_idx = end_idx
            end_idx = self.fused_out_dim
            nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
            nn.init.zeros_(fused_bias[start_idx:end_idx])
            
            # Store initialized weights
            self.W_fused.weight.data = fused_weight
            self.W_fused.bias.data = fused_bias
    
    def compute_dpg_gamma(
        self,
        z_t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DPG mekanizması ile dinamik γ_t hesaplama.
        
        Formül: γ_t = σ(W_γ,up · ReLU(W_γ,down · z_t))
        
        Args:
            z_t: Input tensor [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context for modulation (future extension)
        
        Returns:
            gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
        """
        if not self.use_dpg:
            # Fallback to original MDI approach
            return self.mdi.compute_decay_only(z_t, context)
        
        # Step 1: Down-projection (D -> 128)
        # z_t: [batch, seq_len, model_dim]
        # W_γ,down: [model_dim, dpg_rank]
        # Output: [batch, seq_len, dpg_rank]
        z_projected_down = self.W_gamma_down(z_t)
        
        # Step 2: ReLU activation (non-linearity + non-negativity)
        z_activated = self.dpg_activation(z_projected_down)
        # z_activated: [batch, seq_len, dpg_rank]
        
        # Step 3: Up-projection (128 -> D)
        # W_γ,up: [dpg_rank, model_dim]
        # Output: [batch, seq_len, model_dim]
        z_projected_up = self.W_gamma_up(z_activated)
        
        # Step 4: Sigmoid activation (ensure γ_t ∈ [0, 1])
        gamma = self.dpg_sigmoid(z_projected_up)
        # gamma: [batch, seq_len, model_dim]
        
        # Step 5: Clamp to prevent numerical issues
        # Range: [1e-6, 1-1e-6] to avoid extreme values
        gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
        
        # Optional: Context modulation (if provided)
        if context is not None:
            # Future extension: context-dependent modulation
            # For now, just return gamma
            pass
        
        return gamma
    
    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState,
        hds: Optional[HierarchicalDataStructure] = None,
        use_checkpointing: Optional[bool] = None,
        return_auxiliary_loss: bool = False,
        router_threshold: float = 0.0
    ) -> Tuple[torch.Tensor, MemoryState, Optional[torch.Tensor]]:
        """
        Forward pass through MM-Rec block.
        ...
        router_threshold: Confidence threshold for Sparse LSH Router (default: 0.0)
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
        
        # ========================================================================
        # HEM: Fused Kernel - Single Matmul for QKVZ + PE
        # ========================================================================
        if self.use_hem:
            # Step 1: Normalize input
            x_norm_all = self.norm1(x)  # [batch, seq_len, model_dim]
            
            # Step 2: CRITICAL - Single fused matmul
            # This replaces 6 separate matmul operations with 1
            import torch.nn.functional as F
            fused_output = F.linear(
                x_norm_all,
                self.W_fused.weight,  # [fused_out_dim, model_dim]
                self.W_fused.bias      # [fused_out_dim]
            )
            # fused_output: [batch, seq_len, fused_out_dim]
            
            # Step 3: Split fused output into individual projections
            q_all, k_all, v_all, z_all, p_all, e_all = torch.split(
                fused_output,
                split_size_or_sections=[
                    self.proj_dims['Q'],
                    self.proj_dims['K'],
                    self.proj_dims['V'],
                    self.proj_dims['Z'],
                    self.proj_dims['P'],
                    self.proj_dims['E']
                ],
                dim=-1
            )
            # q_all, k_all, v_all, z_all: [batch, seq_len, model_dim]
            # p_all: [batch, seq_len, pe_dim]
            # e_all: [batch, seq_len, model_dim]
            
            # Step 4: Add positional encoding to input
            x_with_pe = x_norm_all + e_all  # [batch, seq_len, model_dim]
        else:
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
                
                x_with_pe = x_norm_all  # No positional encoding in non-HEM mode
        
        # Sequential processing: Loop over sequence steps
        
        # 1. Pre-compute Gamma for all steps (Parallel)
        # Gamma depends only on z_all (and context k_all), so we can compute it massively parallel
        # This replaces the inside-loop gamma computation
        if self.use_dpg:
            # Use DPG for dynamic gamma computation
            gamma_all = self.compute_dpg_gamma(z_all, context=k_all)
        else:
            # Original MDI approach
            gamma_all = self.mdi.compute_decay_only(z_all, context=k_all)
            
        # 2. Sequential Recurrence using JIT Kernel
        # The core recurrence h_t = ... is sequential due to W_g(h_{t-1})
        # We use a fused JIT kernel to execute this loop efficiently without Python overhead
        # and without redundant kernel launches.
        from .jit_kernels import mm_rec_recurrence_fused
        
        output, h_final = mm_rec_recurrence_fused(
            z_all=z_all,
            gamma_all=gamma_all,
            gate_block_weight=self.W_g.weight,
            gate_block_bias=self.W_g.bias,
            gate_mdi_weight=self.mdi.W_g.weight,
            gate_mdi_bias=self.mdi.W_g.bias,
            h_init=h_prev,  # [batch, 1, model_dim]
            use_mdi_gate=True
        )
        # Update h_prev for next chunk boundaries (if chunking is used)
        h_prev = h_final
        
        # 3. Post-Recurrence Processing (Vectorized)
        
        # Output from recurrence is [batch, seq_len, model_dim] aka h_t sequence
        h_sequence = output
        
        # Step 7: Multi-Memory Attention (Parallel over sequence)
        # We process the entire sequence at once.
        # Note: q_all, v_all are already [batch, seq_len, dim]
        
        mem_context = self.multi_mem_attention(h_sequence, hds, state, q_input=q_all)
        
        # Residual Connection
        # h_attended = h_t + attention + v*0.1
        # v_all * 0.1 is broadcasting 0.1
        v_contribution = v_all * 0.1
        h_attended = h_sequence + mem_context + v_contribution
        
        # Step 8: Dropout & Residual
        # x_residual = x + dropout(h_attended)
        # Note: 'x' is the input to the block. 
        # In HEM/Kernel fusion, 'x' [batch, seq_len, dim] is available.
        x_residual = x + self.dropout_layer(h_attended)
        
        # Step 9: Feed-Forward Network (Parallel)
        x_norm2 = self.norm2(x_residual)
        
        if hasattr(self.ffn, 'router'):
             # Sparse FFN likely supports sequence input too
            ffn_out = self.ffn(x_norm2, router_threshold=router_threshold)
        else:
            ffn_out = self.ffn(x_norm2)
            
        final_output = x_residual + ffn_out
        
        # Step 10: Store Output & Update State
        # 'final_output' is [batch, seq_len, dim]
        output = final_output
        
        # Step 11: State Update (Sequential but fast)
        # We still need to update the state step-by-step for the next block/chunk.
        # Since this is just memory copy, it's fast.
        # We can optimize this if State supports bulk update, but for now loop is fine
        # as it doesn't involve heavy compute.
        for t in range(seq_len):
            h_t_val = h_sequence[:, t, :] # [batch, dim]
            state.update_state_sequential(bank_type='short', new_k=h_t_val, new_v=h_t_val, step=t)

        
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
        
        # Collect auxiliary loss for this block
        L_Aux_block = None
        if return_auxiliary_loss and hasattr(self, '_auxiliary_losses') and len(self._auxiliary_losses) > 0:
            # Average auxiliary losses over sequence
            L_Aux_block = torch.stack(self._auxiliary_losses).mean()
            # Clear for next forward pass
            self._auxiliary_losses = []
        
        if return_auxiliary_loss:
            return output, state, L_Aux_block
        else:
            return output, state

