"""
Quantization Support for MM-Rec Models

Provides quantization-aware training (QAT) and post-training quantization
for model deployment and memory efficiency.

Quantization Types:
1. Dynamic Quantization: Quantize at inference time
2. Static Quantization: Calibration-based quantization
3. QAT (Quantization-Aware Training): Train with quantization simulation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings

# Try to import modern quantization API (PyTorch 1.9+)
try:
    from torch.ao.quantization import (
        QConfig,
        QConfigMapping,
        get_default_qconfig,
        prepare_qat,
        convert,
        prepare,
        ObserverBase,
        FakeQuantizeBase
    )
    _has_modern_quantization = True
except ImportError:
    try:
        from torch.quantization import (
            QConfig,
            default_qconfig,
            prepare_qat,
            convert,
            prepare
        )
        _has_modern_quantization = False
    except ImportError:
        _has_modern_quantization = False
        warnings.warn("PyTorch quantization not available. Install PyTorch 1.9+ for quantization support.")


def get_qat_qconfig(backend: str = 'fbgemm') -> Optional[QConfig]:
    """
    Get QAT (Quantization-Aware Training) QConfig.
    
    Args:
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
    
    Returns:
        QConfig for QAT, or None if quantization not available
    """
    if not _has_modern_quantization:
        return None
    
    try:
        if backend == 'fbgemm':
            from torch.ao.quantization.qconfig import default_qat_qconfig
            return default_qat_qconfig
        elif backend == 'qnnpack':
            from torch.ao.quantization.qconfig import default_qat_qconfig_qnnpack
            return default_qat_qconfig_qnnpack
        else:
            from torch.ao.quantization.qconfig import default_qat_qconfig
            return default_qat_qconfig
    except ImportError:
        return None


def prepare_model_for_qat(
    model: nn.Module,
    backend: str = 'fbgemm',
    inplace: bool = False
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    This inserts fake quantization modules into the model to simulate
    quantization during training. The model learns to adapt to quantization.
    
    Args:
        model: PyTorch model to prepare
        backend: Quantization backend ('fbgemm' or 'qnnpack')
        inplace: Whether to modify model in-place
    
    Returns:
        Model prepared for QAT
    """
    if not _has_modern_quantization:
        raise RuntimeError("Quantization not available. Install PyTorch 1.9+")
    
    if not inplace:
        model = model.copy() if hasattr(model, 'copy') else model
    
    # Get QAT QConfig
    qconfig = get_qat_qconfig(backend)
    if qconfig is None:
        raise RuntimeError(f"QAT QConfig not available for backend: {backend}")
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Prepare model for QAT
    # This inserts FakeQuantize modules
    model.qconfig = qconfig
    
    # Prepare QAT
    try:
        model_prepared = prepare_qat(model, inplace=inplace)
    except Exception as e:
        # Fallback to manual preparation
        warnings.warn(f"Automatic QAT preparation failed: {e}. Using manual preparation.")
        model_prepared = _prepare_model_manual_qat(model, qconfig, inplace)
    
    return model_prepared


def _prepare_model_manual_qat(
    model: nn.Module,
    qconfig: QConfig,
    inplace: bool = False
) -> nn.Module:
    """
    Manually prepare model for QAT by setting qconfig on modules.
    
    Args:
        model: Model to prepare
        qconfig: QConfig to use
        inplace: Whether to modify in-place
    
    Returns:
        Model with QConfig set
    """
    if not inplace:
        model = model.copy() if hasattr(model, 'copy') else model
    
    # Set qconfig on quantizable modules
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.qconfig = qconfig
    
    return model


def convert_to_quantized(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Convert QAT model to quantized model for inference.
    
    This converts FakeQuantize modules to actual quantized operations.
    The model is ready for deployment with INT8 weights.
    
    Args:
        model: QAT-prepared model
        inplace: Whether to modify in-place
    
    Returns:
        Quantized model (INT8 weights)
    """
    if not _has_modern_quantization:
        raise RuntimeError("Quantization not available")
    
    if not inplace:
        model = model.copy() if hasattr(model, 'copy') else model
    
    # Convert to quantized
    model_quantized = convert(model, inplace=inplace)
    
    return model_quantized


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Quantizes weights to INT8, activations quantized at runtime.
    Good for LSTM/RNN models.
    
    Args:
        model: Model to quantize
        dtype: Quantization dtype (torch.qint8 or torch.float16)
    
    Returns:
        Dynamically quantized model
    """
    if not _has_modern_quantization:
        try:
            from torch.quantization import quantize_dynamic
            return quantize_dynamic(model, {nn.Linear}, dtype=dtype)
        except ImportError:
            raise RuntimeError("Dynamic quantization not available")
    
    try:
        from torch.ao.quantization import quantize_dynamic
        return quantize_dynamic(model, {nn.Linear}, dtype=dtype)
    except ImportError:
        raise RuntimeError("Dynamic quantization not available")


def get_model_size_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: Model to measure
        dtype: Data type to use for calculation
    
    Returns:
        Model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.tensor(0, dtype=dtype).element_size()
    size_mb = (total_params * bytes_per_param) / (1024 ** 2)
    return size_mb


def compare_model_sizes(
    model_fp32: nn.Module,
    model_quantized: nn.Module
) -> Dict[str, float]:
    """
    Compare sizes of FP32 and quantized models.
    
    Args:
        model_fp32: Original FP32 model
        model_quantized: Quantized model
    
    Returns:
        Dictionary with size comparisons
    """
    size_fp32 = get_model_size_mb(model_fp32, torch.float32)
    size_quantized = get_model_size_mb(model_quantized, torch.qint8)
    
    return {
        'fp32_size_mb': size_fp32,
        'quantized_size_mb': size_quantized,
        'compression_ratio': size_fp32 / size_quantized if size_quantized > 0 else 0,
        'memory_savings_percent': (1 - size_quantized / size_fp32) * 100 if size_fp32 > 0 else 0
    }


def is_quantization_available() -> bool:
    """Check if quantization is available."""
    return _has_modern_quantization


def get_quantization_backends() -> list:
    """Get available quantization backends."""
    backends = []
    try:
        if hasattr(torch.backends, 'quantized'):
            if hasattr(torch.backends.quantized, 'supported_engines'):
                backends = torch.backends.quantized.supported_engines
            else:
                # Fallback: check common backends
                for backend in ['fbgemm', 'qnnpack']:
                    try:
                        torch.backends.quantized.engine = backend
                        backends.append(backend)
                    except:
                        pass
    except:
        pass
    return backends if backends else []

