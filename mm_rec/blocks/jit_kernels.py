
import torch
from typing import Tuple, List, Optional

@torch.jit.script
def mm_rec_recurrence_fused(
    z_all: torch.Tensor,
    gamma_all: torch.Tensor,
    gate_block_weight: torch.Tensor,
    gate_block_bias: torch.Tensor,
    gate_mdi_weight: torch.Tensor,
    gate_mdi_bias: torch.Tensor,
    h_init: torch.Tensor,
    use_mdi_gate: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled fused recurrence for MM-Rec.
    
    Implements the loop:
    for t in range(seq_len):
        1. MDI Logic:
           gate_mdi = sigmoid(Linear([z_t, h_prev]))
           h_tilde = (1-gate_mdi)*h_prev + gate_mdi*z_t
           h_new = h_tilde + gamma_t * h_prev
           
        2. Block Logic:
           gate_block = sigmoid(Linear(h_prev))
           h_t = z_t * gate_block + gamma_t * h_prev
           h_t += 0.1 * h_new  # Gradient mixing
           
    Args:
        z_all: [batch, seq_len, model_dim]
        gamma_all: [batch, seq_len, model_dim] (Pre-computed)
        gate_block_weight: [model_dim, model_dim]
        gate_block_bias: [model_dim]
        gate_mdi_weight: [model_dim, 2*model_dim]
        gate_mdi_bias: [model_dim]
        h_init: [batch, 1, model_dim]
        
    Returns:
        output_sequence: [batch, seq_len, model_dim]
        final_state: [batch, 1, model_dim]
    """
    batch_size, seq_len, model_dim = z_all.shape
    
    # Pre-allocate output
    output = torch.zeros_like(z_all)
    
    # Initial state (squeeze for computation)
    h_prev = h_init.squeeze(1) # [batch, model_dim]
    
    for t in range(seq_len):
        z_t = z_all[:, t, :]
        gamma_t = gamma_all[:, t, :]
        
        # -----------------------------------------------------------
        # 1. MDI Logic (Inlined)
        # -----------------------------------------------------------
        if use_mdi_gate:
            # cat([z_t, h_prev], dim=-1)
            # Optimization: F.linear(cat) is equivalent to F.linear(z) + F.linear(h)
            # But here we just concat as it's cleaner in JIT
            concat_input = torch.cat([z_t, h_prev], dim=-1)
            gate_mdi = torch.sigmoid(torch.nn.functional.linear(concat_input, gate_mdi_weight, gate_mdi_bias))
            
            h_tilde = (1.0 - gate_mdi) * h_prev + gate_mdi * z_t
            h_new = h_tilde + gamma_t * h_prev
        else:
            # Fallback if no MDI gate (identity)
            h_new = z_t + gamma_t * h_prev
            
        # -----------------------------------------------------------
        # 2. Block Logic
        # -----------------------------------------------------------
        # gate_signal = sigmoid(W_g(h_prev))
        gate_block = torch.sigmoid(torch.nn.functional.linear(h_prev, gate_block_weight, gate_block_bias))
        
        # h_t = z_t * gate_block + gamma * h_prev
        # Note: Original code used 'gamma_new_t' from MDI, which is 'gamma_t' here
        term1 = z_t * gate_block
        term2 = gamma_t * h_prev
        h_t_val = term1 + term2
        
        # -----------------------------------------------------------
        # 3. Combine
        # -----------------------------------------------------------
        # h_t = h_t + 0.1 * h_new_t
        h_t_val = h_t_val + 0.1 * h_new
        
        # Store
        output[:, t, :] = h_t_val
        
        # Update for next step
        h_prev = h_t_val
        
    return output, h_prev.unsqueeze(1)
