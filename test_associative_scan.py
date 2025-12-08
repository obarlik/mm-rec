#!/usr/bin/env python3
"""
Test script for MM-Rec Associative Scan Exponential
Works with or without CUDA
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} found")
except ImportError:
    print("✗ PyTorch not found. Please install: pip install torch")
    sys.exit(1)

try:
    import triton
    print(f"✓ Triton found")
    TRITON_AVAILABLE = True
except ImportError:
    print("⚠ Triton not found. Will use CPU fallback.")
    TRITON_AVAILABLE = False

from mm_rec.core.associative_scan_triton import (
    associative_scan_exponential,
    associative_scan_exponential_cpu_fallback,
    test_associative_scan_correctness,
    test_gradient_correctness,
)

if __name__ == "__main__":
    print("=" * 60)
    print("MM-Rec Associative Scan Exponential - Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n🔧 System Info:")
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Triton available: {TRITON_AVAILABLE}")
    
    # Determine which implementation to use
    use_cpu_fallback = not (cuda_available and TRITON_AVAILABLE)
    
    if use_cpu_fallback:
        print("\n⚠ Note: Using CPU fallback implementation")
        print("   For GPU acceleration, install: pip install triton")
    
    # Run tests
    print("\n" + "=" * 60)
    print("Test 1: Forward Pass Correctness")
    print("=" * 60)
    test1_passed = test_associative_scan_correctness(use_cpu_fallback=use_cpu_fallback)
    
    print("\n" + "=" * 60)
    print("Test 2: Gradient Computation")
    print("=" * 60)
    test2_passed = test_gradient_correctness(use_cpu_fallback=use_cpu_fallback)
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print(f"  Forward test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Gradient test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Check output above for details.")
        sys.exit(1)

