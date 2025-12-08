"""
MM-Rec Core Components
"""

try:
    from .associative_scan_triton import (
        associative_scan_exponential,
        AssociativeScanExponential,
        associative_scan_exponential_cpu_fallback,
        test_associative_scan_correctness,
        test_gradient_correctness,
    )
    __all__ = [
        'associative_scan_exponential',
        'AssociativeScanExponential',
        'associative_scan_exponential_cpu_fallback',
        'test_associative_scan_correctness',
        'test_gradient_correctness',
    ]
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings
    warnings.warn(f"Could not import associative_scan_triton: {e}")
    __all__ = []

