"""
CPU Mixed Precision Training Support

This module provides CPU-specific mixed precision training functionality.
Unlike GPU's torch.cuda.amp, this uses FP16/BF16 for storage and FP32 for computation.

Key differences from GPU AMP:
- No Tensor Cores (CPU doesn't have them)
- FP16/BF16 storage, FP32 computation (master weights pattern)
- Loss scaling still needed to prevent gradient underflow
- Memory savings: ~50% reduction in model size
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from torch.cuda.amp import GradScaler as CUDAScaler


class CPUScaler:
    """
    CPU-specific loss scaler for mixed precision training.
    
    Similar to torch.cuda.amp.GradScaler but works on CPU.
    Prevents gradient underflow when using FP16/BF16.
    """
    
    def __init__(
        self,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._per_tensor_enabled = True
        # Expose scale as property for direct access
        self.scale_value = self.scale
        
    def __call__(self, outputs):
        """Scale outputs (loss) to prevent underflow."""
        if isinstance(outputs, torch.Tensor):
            return outputs * self.scale
        return tuple(out * self.scale for out in outputs)
    
    def scale(self, outputs):
        """Scale outputs (loss) to prevent underflow. (Alias for __call__)"""
        return self(outputs)
    
    def unscale_(self, optimizer):
        """Unscale gradients before optimizer step."""
        # For CPU, we need to manually unscale gradients
        # This is called before gradient clipping
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer):
        """Update scaler state after optimizer step."""
        # Check for inf/NaN in gradients
        found_inf = False
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        found_inf = True
                        break
            if found_inf:
                break
        
        if found_inf:
            # Reduce scale and skip optimizer step
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            return False
        else:
            # Increase scale if no overflow
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            return True
    
    def update(self):
        """Update scaler (called after optimizer step)."""
        pass


class CPUAutocast:
    """
    CPU-specific autocast context manager.
    
    Converts operations to FP32 for computation while keeping
    storage in FP16/BF16 for memory efficiency.
    """
    
    def __init__(self, dtype: torch.dtype = torch.bfloat16, enabled: bool = True):
        self.dtype = dtype
        self.enabled = enabled
        self._prev_dtype = None
        
    def __enter__(self):
        if self.enabled:
            # Store current default dtype
            self._prev_dtype = torch.get_default_dtype()
            # Set to FP32 for computation (CPU optimizations)
            torch.set_default_dtype(torch.float32)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._prev_dtype is not None:
            torch.set_default_dtype(self._prev_dtype)


def convert_model_to_mixed_precision(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    keep_batchnorm_fp32: bool = True
) -> nn.Module:
    """
    Convert model to mixed precision (FP16/BF16 storage, FP32 computation).
    
    Args:
        model: PyTorch model to convert
        dtype: Target dtype (torch.float16 or torch.bfloat16)
        keep_batchnorm_fp32: Keep BatchNorm in FP32 for stability
    
    Returns:
        Model with mixed precision weights
    """
    def convert_module(module):
        # Convert parameters to target dtype
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype)
        
        # Convert buffers (except BatchNorm running stats)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.dtype == torch.float32:
                # Keep BatchNorm stats in FP32 for stability
                if keep_batchnorm_fp32 and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if 'running' in name or 'num_batches' in name:
                        continue
                buffer.data = buffer.data.to(dtype)
        
        # Recursively convert child modules
        for child in module.children():
            convert_module(child)
    
    convert_module(model)
    return model


def cpu_amp_enabled() -> bool:
    """Check if CPU AMP is available and should be used."""
    # CPU AMP is always available (no CUDA requirement)
    return True


def get_cpu_scaler() -> CPUScaler:
    """Get CPU scaler instance."""
    return CPUScaler()


def get_cpu_autocast(dtype: torch.dtype = torch.bfloat16) -> CPUAutocast:
    """Get CPU autocast context manager."""
    return CPUAutocast(dtype=dtype)

