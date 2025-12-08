"""
MM-Rec Memory Decay/Integration (MDI)
Implements gated memory integration with learnable decay coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MemoryDecayIntegration(nn.Module):
    """
    Memory Decay/Integration mechanism for MM-Rec.
    
    Implements the core formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    
    This module manages:
    - Gated integration between new input (z_t) and previous state (h_{t-1})
    - Learnable decay coefficients (γ)
    - Context-dependent modulation
    
    Args:
        model_dim: Model dimension (hidden_dim)
        inner_dim: Inner dimension for decay coefficient computation
        use_context_modulation: Whether to use context-dependent modulation
    """
    
    def __init__(
        self,
        model_dim: int,
        inner_dim: Optional[int] = None,
        use_context_modulation: bool = True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
        self.use_context_modulation = use_context_modulation
        
        # Gating weight: Controls how much of new (z_t) vs old (h_{t-1}) to combine
        # Input: concatenated [z_t, h_{t-1}] -> output: gate value
        self.W_g = nn.Linear(model_dim * 2, model_dim)
        
        # Decay weight: Learns decay coefficient γ
        # Input: z_t -> output: decay coefficient
        self.W_gamma = nn.Sequential(
            nn.Linear(model_dim, self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, model_dim),
            nn.Sigmoid()  # Ensure γ is in [0, 1]
        )
        
        # Context modulation (optional): Makes integration context-dependent
        if self.use_context_modulation:
            self.W_context = nn.Sequential(
                nn.Linear(model_dim, self.inner_dim),
                nn.GELU(),
                nn.Linear(self.inner_dim, model_dim),
                nn.Sigmoid()
            )
        else:
            self.W_context = None
    
    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Compute new state h_t and decay coefficient γ.
        
        Implements: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        
        Args:
            z_t: New input [batch, seq_len, model_dim] or [batch, model_dim]
            h_prev: Previous hidden state [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context tensor for modulation [batch, seq_len, model_dim] or [batch, model_dim]
        
        Returns:
            Tuple of (h_new, gamma):
                - h_new: New hidden state [batch, seq_len, model_dim] or [batch, model_dim]
                - gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
        """
        # Ensure same shape
        assert z_t.shape == h_prev.shape, f"Shape mismatch: z_t {z_t.shape} vs h_prev {h_prev.shape}"
        
        # Compute gating signal: g = σ(W_g · [z_t, h_prev])
        # Concatenate z_t and h_prev along last dimension
        concat_input = torch.cat([z_t, h_prev], dim=-1)  # [..., model_dim * 2]
        gate = torch.sigmoid(self.W_g(concat_input))  # [..., model_dim]
        
        # Gated integration: h_tilde = (1 - g) ⊙ h_prev + g ⊙ z_t
        h_tilde = (1 - gate) * h_prev + gate * z_t
        
        # Compute decay coefficient: γ = σ(W_γ · z_t)
        gamma = self.W_gamma(z_t)  # [..., model_dim]
        
        # Apply context modulation if enabled
        if self.use_context_modulation and context is not None:
            modulation = self.W_context(context)  # [..., model_dim]
            gamma = gamma * modulation
        
        # Clamp decay coefficient to prevent numerical issues
        # Range: [1e-6, 1-1e-6] to avoid extreme values
        gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
        
        # Apply decay: h_new = h_tilde + γ ⊙ h_prev (residual-like)
        # This approximates: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        h_new = h_tilde + gamma * h_prev
        
        return h_new, gamma
    
    def compute_decay_only(
        self,
        z_t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute only decay coefficient γ without state update.
        
        Useful for Associative Scan where we need γ values separately.
        
        Args:
            z_t: New input [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context tensor
        
        Returns:
            gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
        """
        gamma = self.W_gamma(z_t)
        
        if self.use_context_modulation and context is not None:
            modulation = self.W_context(context)
            gamma = gamma * modulation
        
        gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
        return gamma

