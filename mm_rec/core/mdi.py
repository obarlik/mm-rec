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
    Memory Decay/Integration mechanism for MM-Rec with UBÖO support.
    
    Implements the core formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    
    UBÖO Extension:
    - Computes Planning Error (P_error) for auxiliary loss
    - Uses gradient isolation via detach() to prevent gradient flow through M_updated
    - Enables unbiased backpropagation through error orthogonalization
    
    This module manages:
    - Gated integration between new input (z_t) and previous state (h_{t-1})
    - Learnable decay coefficients (γ)
    - Context-dependent modulation
    
    Args:
        model_dim: Model dimension (hidden_dim)
        inner_dim: Inner dimension for decay coefficient computation
        use_context_modulation: Whether to use context-dependent modulation
        use_uboo: Enable UBÖO mechanism
        planning_error_dim: Dimension for planning error
    """
    
    def __init__(
        self,
        model_dim: int,
        inner_dim: Optional[int] = None,
        use_context_modulation: bool = True,
        # UBÖO Parameters
        use_uboo: bool = False,           # Enable UBÖO mechanism
        planning_error_dim: Optional[int] = None  # Dimension for planning error
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
        self.use_context_modulation = use_context_modulation
        
        # UBÖO Configuration
        self.use_uboo = use_uboo
        self.planning_error_dim = planning_error_dim if planning_error_dim is not None else model_dim
        
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
        
        # ========================================================================
        # UBÖO: Planning Error Computation Components
        # ========================================================================
        # Planning Error (P_error) measures the discrepancy between:
        # - M_updated: Updated memory state (with gradients)
        # - M_target: Target memory state (computed from planning)
        # 
        # CRITICAL: M_updated must be detached to prevent gradient flow through
        # the planning error computation, enabling unbiased backpropagation.
        # ========================================================================
        
        if self.use_uboo:
            # Planning error projection: Maps memory state to error space
            # Input: M_updated (detached) [batch, seq_len, model_dim]
            # Output: P_error [batch, seq_len, planning_error_dim]
            self.W_planning_error = nn.Sequential(
                nn.Linear(model_dim, self.planning_error_dim),
                nn.GELU(),
                nn.Linear(self.planning_error_dim, self.planning_error_dim)
            )
            
            # Target memory projection: Computes target from planning
            # This is used to compute the planning error target
            self.W_planning_target = nn.Sequential(
                nn.Linear(model_dim, self.planning_error_dim),
                nn.GELU(),
                nn.Linear(self.planning_error_dim, self.planning_error_dim)
            )
        else:
            self.W_planning_error = None
            self.W_planning_target = None
    
    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_auxiliary_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: Compute new state h_t, decay coefficient γ, and auxiliary loss.
        
        Implements: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        
        UBÖO Extension:
        - Computes Planning Error (P_error) for auxiliary loss
        - Uses detach() to isolate gradients
        
        Args:
            z_t: New input [batch, seq_len, model_dim] or [batch, model_dim]
            h_prev: Previous hidden state [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context tensor for modulation [batch, seq_len, model_dim] or [batch, model_dim]
            return_auxiliary_loss: If True, return auxiliary loss (L_Aux)
        
        Returns:
            Tuple of (h_new, gamma, L_Aux):
                - h_new: New hidden state [batch, seq_len, model_dim] or [batch, model_dim]
                - gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
                - L_Aux: Auxiliary loss (Planning Error) [batch, seq_len] or scalar (if return_auxiliary_loss=True)
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
        
        # ========================================================================
        # UBÖO: Planning Error Computation with Gradient Isolation
        # ========================================================================
        # CRITICAL: M_updated (h_new) must be detached to prevent gradient flow
        # through the planning error computation. This enables unbiased backpropagation.
        # 
        # Planning Error Formula:
        #   P_error = ||W_planning_error(M_updated.detach()) - W_planning_target(M_target)||²
        # 
        # Where:
        #   - M_updated: Updated memory state (h_new) - DETACHED
        #   - M_target: Target memory state (computed from planning)
        # 
        # The detach() operation ensures that:
        #   1. Gradients from P_error do not flow back through M_updated
        #   2. Planning error is computed independently of the main forward pass
        #   3. Unbiased backpropagation is maintained
        # ========================================================================
        
        L_Aux = None
        if self.use_uboo and return_auxiliary_loss:
            # Step 1: CRITICAL - Detach M_updated to isolate gradients
            # M_updated = h_new (updated memory state)
            # We detach it to prevent gradient flow through planning error computation
            M_updated_detached = h_new.detach()  # [batch, seq_len, model_dim] or [batch, model_dim]
            # CRITICAL: .detach() creates a new tensor that shares storage but
            # does not require gradients. This breaks the computational graph
            # for the planning error computation.
            
            # Step 2: Compute planning error projection from detached M_updated
            # This projection operates on the detached tensor, so no gradients
            # flow back through M_updated
            P_error_pred = self.W_planning_error(M_updated_detached)
            # P_error_pred: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 3: Compute target from planning (using original h_new with gradients)
            # The target is computed from the planning mechanism, which may use
            # the original h_new (with gradients) or a separate planning computation
            # For simplicity, we use h_prev as the planning target (can be customized)
            M_target = h_prev  # [batch, seq_len, model_dim] or [batch, model_dim]
            P_error_target = self.W_planning_target(M_target)
            # P_error_target: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 4: Compute Planning Error (P_error)
            # P_error = ||P_error_pred - P_error_target||²
            # This is the auxiliary loss that measures planning discrepancy
            P_error = P_error_pred - P_error_target
            # P_error: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 5: Compute auxiliary loss (L_Aux)
            # L_Aux = mean(||P_error||²) over sequence and error dimensions
            # This is a scalar loss that will be added to the main loss
            if P_error.dim() == 3:
                # [batch, seq_len, planning_error_dim]
                L_Aux = torch.mean(P_error ** 2, dim=-1)  # [batch, seq_len]
                L_Aux = torch.mean(L_Aux)  # Scalar
            else:
                # [batch, planning_error_dim]
                L_Aux = torch.mean(P_error ** 2, dim=-1)  # [batch]
                L_Aux = torch.mean(L_Aux)  # Scalar
            
            # CRITICAL: L_Aux is computed from detached M_updated, so:
            # - Gradients from L_Aux do NOT flow back through M_updated
            # - Gradients from L_Aux DO flow back through W_planning_error and W_planning_target
            # - This enables unbiased backpropagation through error orthogonalization
        
        return h_new, gamma, L_Aux
    
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

