
"""
MM-Rec Associative Scan (Exponential Product) - PyTorch Native GPU Implementation
Replaces Triton kernel for Windows/Non-Linux compatibility while maintaining GPU acceleration.
"""

import torch
import torch.nn.functional as F

class AssociativeScanExponential(torch.autograd.Function):
    """
    PyTorch native implementation of associative scan using torch.cumsum.
    This provides O(N) parallel scan on GPU without custom Triton kernels.
    
    Implements: Y_t = ∏_{i=1}^t γ_i using Log-Sum-Exp pattern.
    """
    
    @staticmethod
    def forward(ctx, gamma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Compute cumulative exponential product using torch.cumsum.
        
        Args:
            gamma: [BATCH, HEADS, SEQ_LEN, D_HEAD] decay coefficients
            
        Returns:
            cumulative_product: [BATCH, HEADS, SEQ_LEN, D_HEAD]
        """
        ctx.save_for_backward(gamma)
        
        # Convert to FP32 for numerical stability in log-space
        gamma_fp32 = gamma.to(torch.float32)
        
        # 1. Convert to log-space: log(gamma)
        # Add epsilon to prevent log(0)
        epsilon = 1e-8
        # Clamp gamma to strict (0, 1] range to ensure log <= 0
        gamma_fp32 = torch.clamp(gamma_fp32, min=0.0, max=1.0)
        log_gamma = torch.log(gamma_fp32 + epsilon)
        
        # 2. Cumulative sum in log-space (equivalent to cumulative product in linear space)
        # torch.cumsum is highly optimized on GPU
        log_cumsum = torch.cumsum(log_gamma, dim=2)
        
        # 3. Convert back to linear space
        # Use simple exp() since we are just doing products
        # For very long sequences, we might need more stability, but for <32k this is usually fine
        # Clamp to avoid potential overflow (though unlikely with gamma <= 1)
        log_cumsum = torch.clamp(log_cumsum, max=0.0)
        cumulative_product = torch.exp(log_cumsum)
        
        # 4. Convert back to original dtype
        return cumulative_product.to(gamma.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for cumulative product.
        
        dY/dgamma = ?
        Use the property: Y_t = gamma_1 * ... * gamma_t
        dY_t / dgamma_i = Y_t / gamma_i (for i <= t)
        
        Total gradient dL/dgamma_i = sum_t (dL/dY_t * dY_t/dgamma_i)
                                 = sum_{t=i}^T (grad_output_t * Y_t / gamma_i)
                                 = (1/gamma_i) * sum_{t=i}^T (grad_output_t * Y_t)
                                 
        The term sum_{t=i}^T (...) is a reverse cumulative sum.
        """
        gamma, = ctx.saved_tensors
        
        # Recompute Y_t (cumulative product) efficiently
        # We could save it, but recomputing saves memory
        gamma_fp32 = gamma.to(torch.float32)
        epsilon = 1e-8
        log_gamma = torch.log(gamma_fp32 + epsilon)
        log_cumsum = torch.cumsum(log_gamma, dim=2)
        Y_t = torch.exp(log_cumsum)
        
        # Compute term to scan: grad_output * Y_t
        grad_output_fp32 = grad_output.to(torch.float32)
        term_to_scan = grad_output_fp32 * Y_t
        
        # Compute reverse cumulative sum
        # torch.flip -> cumsum -> torch.flip
        reverse_cumsum = torch.flip(torch.cumsum(torch.flip(term_to_scan, [2]), dim=2), [2])
        
        # Final gradient: reverse_cumsum / gamma
        grad_gamma = reverse_cumsum / (gamma_fp32 + epsilon)
        
        return grad_gamma.to(gamma.dtype)

def associative_scan_exponential_torch(gamma: torch.Tensor) -> torch.Tensor:
    """
    User-facing function acting as drop-in replacement for Triton version.
    """
    return AssociativeScanExponential.apply(gamma)
