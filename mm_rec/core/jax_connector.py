
"""
JAX Connector for MM-Rec
Provides zero-copy data transfer between PyTorch and JAX using DLPack.
Implements the optimized associative scan using JAX's XLA compiler, wrapped in a PyTorch Autograd Function.
"""

import torch
import jax
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack
from torch.autograd import Function
import functools

# Define the JAX scan function (Forward)
@functools.partial(jax.jit, static_argnames=['axis'])
def jax_scan_kernel_fwd(x, axis=1):
    def mul_op(a, b): return a * b
    return jax.lax.associative_scan(mul_op, x, axis=axis)

# Define JAX reverse scan function (Backward)
# For grad_gamma calculation: sum(grad_output[t:] * output[t:] / gamma[t] ...)
# This is complex. Alternatively, use JAX's VJP (Vector-Jacobian Product).
@functools.partial(jax.jit, static_argnames=['axis'])
def jax_scan_vjp(gamma, grad_output, axis=1):
    """
    Compute VJP for associative scan (product).
    y = scan(gamma)
    grad_gamma = vjp(y, grad_y)
    """
    def scan_fn(g):
        def mul_op(a, b): return a * b
        return jax.lax.associative_scan(mul_op, g, axis=axis)
    
    # Use JAX's automatic differentiation
    _, vjp_fn = jax.vjp(scan_fn, gamma)
    grad_gamma = vjp_fn(grad_output)[0]
    return grad_gamma

class JaxScanner(Function):
    """
    Autograd Function wrapping JAX associative scan.
    """
    
    @staticmethod
    def forward(ctx, x_torch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: PyTorch -> JAX Scan -> PyTorch
        """
        ctx.device_type = x_torch.device.type
        ctx.shape = x_torch.shape
        
        # 1. Detach for DLPack (Zero-Copy)
        x_detached = x_torch.detach()
        if not x_detached.is_contiguous():
            x_detached = x_detached.contiguous()
            
        # PyTorch -> JAX
        try:
             # Standard DLPack (PyTorch >= 1.10)
             x_jax = jax_dlpack.from_dlpack(x_detached)
        except Exception:
             # Fallback
             dl_cap = torch_dlpack.to_dlpack(x_detached)
             x_jax = jax_dlpack.from_dlpack(dl_cap)
        
        # 2. JAX Compute
        # Determine axis
        axis = -2
        if x_torch.ndim == 3: axis = 1
        elif x_torch.ndim == 4: axis = 2
        
        ctx.axis = axis
        
        res_jax = jax_scan_kernel_fwd(x_jax, axis=axis)
        
        # 3. JAX -> PyTorch
        # JAX arrays implement __dlpack__ protocol in newer versions
        try:
             res_torch = torch_dlpack.from_dlpack(res_jax)
        except Exception:
             # Fallback for older variations or if jax needs explicit export (rare now)
             # Note: jax.dlpack.to_dlpack is removed in recent versions
             res_torch = torch_dlpack.from_dlpack(res_jax) # Try again, usually works
        
        # Save input/output for backward if needed 
        # (JAX VJP might need original input)
        # Note: We save TENSORS, but for JAX VJP we'll need to send them back to JAX in backward.
        ctx.save_for_backward(x_detached)
        
        return res_torch

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: PyTorch Grad -> JAX VJP -> PyTorch Grad
        """
        x_detached, = ctx.saved_tensors
        axis = ctx.axis
        
        # 1. Prepare JAX inputs
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        try:
            g_jax = jax_dlpack.from_dlpack(grad_output)
            x_jax = jax_dlpack.from_dlpack(x_detached)
        except Exception:
            dl_g = torch_dlpack.to_dlpack(grad_output)
            dl_x = torch_dlpack.to_dlpack(x_detached)
            g_jax = jax_dlpack.from_dlpack(dl_g)
            x_jax = jax_dlpack.from_dlpack(dl_x)
            
        # 2. Compute VJP using JAX
        grad_gamma_jax = jax_scan_vjp(x_jax, g_jax, axis=axis)
        
        # 3. Convert back to PyTorch
        grad_gamma_torch = torch_dlpack.from_dlpack(grad_gamma_jax)
        
        return grad_gamma_torch

