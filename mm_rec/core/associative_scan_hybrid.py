"""
Hybrid Associative Scan: JAX + Triton
Combines JAX for CPU and Triton for GPU
"""

import torch
# Optional triton import
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

from typing import Optional
from .associative_scan_triton import AssociativeScanExponential


@triton.jit
def stable_log_sum_exp_fp64(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    """
    Stable log-sum-exp with FP64 precision for numerical stability.
    Used only for log-space accumulation in associative scan.
    """
    max_val = tl.maximum(a, b)
    diff = tl.abs(a - b)
    # Clamp diff to prevent overflow in exp(-diff)
    diff_clamped = tl.minimum(diff, 20.0)  # exp(-20) ≈ 0
    return max_val + tl.log1p(tl.exp(-diff_clamped))


class HybridPrecisionAssociativeScan:
    """
    Hybrid precision associative scan:
    - BF16: Model weights and activations
    - FP64: Log-space cumulative sum (for numerical stability)
    """
    
    @staticmethod
    def forward(gamma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hybrid precision.
        
        Args:
            gamma: Decay coefficients [batch, heads, seq_len, head_dim] in BF16
        
        Returns:
            cumulative_product: [batch, heads, seq_len, head_dim] in BF16
        """
        # Convert gamma to FP32 for log operations
        gamma_fp32 = gamma.to(torch.float32)
        
        # Step 1: Convert to log-space (FP32)
        epsilon = 1e-8
        log_gamma = torch.log(gamma_fp32 + epsilon)
        log_gamma = torch.clamp(log_gamma, min=-50.0, max=0.0)
        
        # Step 2: Convert to FP64 for critical accumulation
        log_gamma_fp64 = log_gamma.to(torch.float64)
        
        # Step 3: Cumulative sum in FP64 (double precision)
        # This is the critical operation that needs FP64
        log_cumsum_fp64 = torch.cumsum(log_gamma_fp64, dim=2)
        
        # Step 4: Convert back to FP32 for exp
        log_cumsum_fp32 = log_cumsum_fp64.to(torch.float32)
        
        # Step 5: Stable exponential conversion
        max_log = torch.max(log_cumsum_fp32, dim=2, keepdim=True)[0]
        stable_log = log_cumsum_fp32 - max_log
        cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
        
        # Step 6: Convert back to original dtype (BF16)
        cumulative_product = cumulative_product.to(gamma.dtype)
        
        return cumulative_product
    
    @staticmethod
    def backward(grad_output: torch.Tensor, gamma: torch.Tensor, cumulative_product: torch.Tensor) -> torch.Tensor:
        """
        Backward pass with hybrid precision.
        
        Args:
            grad_output: Gradient w.r.t. output [batch, heads, seq_len, head_dim]
            gamma: Original input [batch, heads, seq_len, head_dim]
            cumulative_product: Forward pass output
        
        Returns:
            grad_gamma: Gradient w.r.t. gamma
        """
        # Convert to FP32 for gradient computation
        grad_output_fp32 = grad_output.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        cumulative_product_fp32 = cumulative_product.to(torch.float32)
        
        # Gradient computation (standard)
        # grad_gamma = grad_output * (cumulative_product / gamma)
        epsilon = 1e-8
        grad_gamma_fp32 = grad_output_fp32 * (cumulative_product_fp32 / (gamma_fp32 + epsilon))
        
        # Convert back to original dtype
        grad_gamma = grad_gamma_fp32.to(gamma.dtype)
        
        return grad_gamma


def associative_scan_exponential_hybrid(gamma: torch.Tensor) -> torch.Tensor:
    """
    User-facing function for hybrid precision associative scan.
    
    Uses:
    - BF16 for model weights/activations
    - FP64 for log-space accumulation (critical for stability)
    
    Args:
        gamma: [batch, heads, seq_len, head_dim] in BF16
    
    Returns:
        cumulative_product: [batch, heads, seq_len, head_dim] in BF16
    """
    return HybridPrecisionAssociativeScan.forward(gamma)

