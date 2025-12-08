#!/usr/bin/env python3
"""
Quick test script - works without CUDA
"""

def main():
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} found")
    except ImportError:
        print("✗ PyTorch not found!")
        print("  Install with: pip install torch")
        return
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    
    # Try to import
    try:
        from mm_rec.core.associative_scan_triton import (
            associative_scan_exponential_cpu_fallback,
            test_associative_scan_correctness,
        )
        print("✓ Module imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Run simple test
    print("\n" + "="*50)
    print("Running Forward Pass Test (CPU Fallback)")
    print("="*50)
    
    try:
        passed = test_associative_scan_correctness(use_cpu_fallback=True)
        if passed:
            print("\n🎉 Test PASSED!")
        else:
            print("\n⚠ Test had some differences (check tolerance)")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

