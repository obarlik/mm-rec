
"""
Sparse MM-Rec Block
Integrates 'Collision-Based Sparse Training' into the core architecture.
Uses LSH/SimHash for fast routing and Chunk-Level Granularity for cache efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional
import math

# Try to import core components (fallback if missing during dev)
try:
    from ..core.associative_scan_triton import associative_scan_exponential
    from ..core.mdi import MemoryDecayIntegration
    from ..core.hds import HierarchicalDataStructure
    from ..core.memory_state import MemoryState
    from .attention import MultiMemoryAttention
    from ..core.jax_connector import JaxScanner
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    pass

class LSHRouter(nn.Module):
    """
    Locality Sensitive Hashing (SimHash) Router.
    Routes input chunks to experts based on geometric similarity.
    
    Verified Features:
    - Chunk-Level Routing (routes groups of L tokens)
    - Top-2 Expert Selection (Mitigates mixed-chunk penalty)
    """
    def __init__(self, dim: int, num_experts: int = 64, num_bits: int = 6):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_bits = num_bits # 2^6 = 64
        
        # Random Hyperplanes (Fixed, not learned - proven effective in demos)
        # Shape: [Dim, Bits]
        self.register_buffer("hyperplanes", torch.randn(dim, num_bits))
        
        # Powers of 2 for converting bits to integer ID: [1, 2, 4, 8, 16, 32]
        self.register_buffer("powers", 2 ** torch.arange(num_bits))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [Batch, NumChunks, ChunkSize, Dim]
            
        Returns:
            expert_indices: [Batch, NumChunks, TopK]
            gates: [Batch, NumChunks, TopK] (Binary 1.0/0.0 for LSH)
        """
        # 1. Mean Pooling per Chunk (Get "Gist" of the chunk)
        # x: [B, N_C, C_S, D] -> chunk_emb: [B, N_C, D]
        chunk_emb = x.mean(dim=2) 
        
        # 2. Hashing (SimHash)
        # Project: [B, N_C, D] @ [D, Bits] -> [B, N_C, Bits]
        projections = torch.matmul(chunk_emb, self.hyperplanes)
        
        # Bitwise Generation: Sign(Proj) -> 0/1
        bits = (projections > 0).int()
        
        # Convert to Expert ID: [B, N_C]
        # Sum(Bits * Powers)
        expert_ids = torch.sum(bits * self.powers, dim=-1)
        
        # 3. Top-2 Logic (Mitigation)
        # Since LSH is deterministic (1 input -> 1 bucket), how do we get Top-2?
        # Strategy: We look for the "Second Best" bucket.
        # The "Second Best" is usually a neighbor in Hamming space (flip 1 bit).
        # Which bit? The one where projection was closest to 0 (the "Boundary" case).
        
        # Find index of the bit with minimum absolute projection value (the "weakest" decision)
        # projections: [B, N_C, Bits]
        abs_proj = torch.abs(projections)
        weakest_bit_idx = torch.argmin(abs_proj, dim=-1) # [B, N_C]
        
        # Create a mask to flip that bit
        # We need 2^weakest_bit_idx
        flip_mask = (2 ** weakest_bit_idx).int()
        
        # Expert 1: Original ID
        expert_1 = expert_ids
        
        # Expert 2: Flip the weakest bit (Neighbor across the closest boundary)
        expert_2 = expert_ids ^ flip_mask
        
        # Stack: [B, N_C, 2]
        expert_indices = torch.stack([expert_1, expert_2], dim=-1)
        
        # Gates: LSH is hard routing, so weight is 1.0 (or 0.5 for averaging)
        # Let's use 1.0 for now, implementation can normalize later if needed.
        gates = torch.ones_like(expert_indices, dtype=x.dtype)
        
        # 4. Confidence Score (For Uncertainty / Hallucination Detection)
        # Measure: Mean absolute distance from the hyperplane (margin)
        # If projections are close to 0, confidence is low.
        # [B, N_C]
        confidence = torch.abs(projections).mean(dim=-1)
        
        return expert_indices, gates, confidence



class SparseFFN(nn.Module):
    """
    Sparse Feed-Forward Network using Chunk-Level LSH Routing.
    Replaces the dense FFN in MM-Rec with Sparse Experts (MoE style).
    """
    
    def __init__(
        self,
        model_dim: int = 4096,
        chunk_size: int = 128,  # Verified mitigation size
        num_experts: int = 64,
        ffn_dim: Optional[int] = None, # The intermediate dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.chunk_size = chunk_size
        self.num_experts = num_experts
        self.ffn_dim = ffn_dim if ffn_dim else model_dim * 4
        
        # Calculate needed bits: log2(num_experts)
        # Ensure num_experts is power of 2
        import math
        bits = int(math.log2(num_experts))
        assert 2**bits == num_experts, "num_experts must be a power of 2 for LSH"
        
        # 1. Router
        self.router = LSHRouter(model_dim, num_experts, num_bits=bits)
        
        # 2. Experts 
        # FFN is usually Up -> Act -> Down
        # For efficiency in this sparse block, let's implement a single Linear Expert for now
        # OR implement the full Up/Down structure if memory allows. 
        # A full Up/Down for 64 experts is heavy (64 * 4096 * 16384 * 2 params).
        # Let's start with a simplified Expert: Linear(D, D) or similar.
        # But wait, standard FFN is D -> 4D -> D.
        # If we have 64 experts, maybe each expert is smaller? 
        # "Chunk-Level Routing" benchmark used simple dense layers.
        # Let's keep it structurally similar to the Dense FFN but partitioned.
        # To avoid massive parameter explosion, let's assume experts share the Up/Down structure
        # but have different weights. 
        # Actually, let's define the expert as a single matrix [D, D] for this v1 integration
        # to match the "Chunk Routing" experiment logic which compared Dense(D,D) to Sparse(D,D).
        # If the user wants full FFN replacement, we might need [NumExperts, D, 4D] and [NumExperts, 4D, D].
        # That's huge. 
        # Compromise: Let's use `expert_dim` = `model_dim` (Linear Expert).
        
        self.expert_weights = nn.Parameter(torch.randn(num_experts, model_dim, model_dim) * 0.02)
        
        # Normalization
        # self.norm = nn.LayerNorm(model_dim) # Usually applied before or after in the block, not inside FFN
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, router_threshold: float = 0.0) -> torch.Tensor:
        """
        x: [Batch, Seq, Dim]
        router_threshold: If confidence < this, output is masked (Creative vs Strict Mode).
                          Default 0.0 (Accept everything).
        """
        B, L, D = x.shape
        
        # Pad sequence if not divisible by chunk_size
        if L % self.chunk_size != 0:
            pad_len = self.chunk_size - (L % self.chunk_size)
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            x_padded = x
            L_padded = L
            
        # 1. Chunking
        # [B, L, D] -> [B, NumChunks, ChunkSize, D]
        num_chunks = L_padded // self.chunk_size
        x_reshaped = x_padded.view(B, num_chunks, self.chunk_size, D)
        
        # 2. Routing
        # indices: [B, N_C, TopK=2]
        # confidence: [B, N_C]
        expert_indices, gates, confidence = self.router(x_reshaped)
        
        # Apply Threshold (Uncertainty Gating)
        if router_threshold > 0.0:
            # Create mask: 1.0 if confident, 0.0 if unsure
            # [B, N_C] -> [B, N_C, 1, 1] for broadcasting
            confidence_mask = (confidence >= router_threshold).float()
            confidence_mask = confidence_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            confidence_mask = 1.0
        
        # 3. Sparse Computation
        output = torch.zeros_like(x_reshaped)
        
        for k in range(2): 
            idx_k = expert_indices[:, :, k] 
            gate_k = gates[:, :, k].unsqueeze(-1).unsqueeze(-1) # [B, N_C, 1, 1]
            
            # Efficient Gather & Compute
            # Flatten to [TotalChunks, ChunkSize, Dim]
            x_flat = x_reshaped.flatten(0, 1) # [B*NC, CS, D]
            idx_flat = idx_k.flatten() # [B*NC]
            
            # Gather weights: [B*NC, D, D]
            # expert_weights: [E, D, D]
            w_gathered = self.expert_weights[idx_flat]
            
            # Matmul: [BatchChunk, ChunkSize, D] @ [BatchChunk, D, D] -> [BatchChunk, ChunkSize, D]
            out_flat = torch.bmm(x_flat, w_gathered)
            
            # Reshape back
            out_k = out_flat.view(B, num_chunks, self.chunk_size, D)
            
            # Apply Gating AND Confidence Mask
            output += out_k * gate_k * confidence_mask
            
        # Normalize
        output = output / 2.0 
        
        # 4. Un-chunk & Dropout
        output = output.view(B, L_padded, D)
        output = self.dropout(output)
        
        # Cut padding
        if L != L_padded:
            output = output[:, :L, :]
            
        return output


