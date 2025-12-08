"""
Memory Debugging Script for MM-Rec
Detects O(N²) memory growth and Triton fallback issues
"""

import torch
import torch.cuda
import sys
import os
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.utils.memory_profiler import MemoryProfiler, profile_memory_growth


def test_triton_fallback():
    """Test if Triton kernel is working or falling back to CPU."""
    print("\n" + "="*80)
    print("🔬 TEST 1: Triton Kernel Fallback Detection")
    print("="*80)
    
    from mm_rec.core.associative_scan_triton import associative_scan_exponential
    
    # Test with different sequence lengths
    test_seq_lengths = [1024, 8192, 32768]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    for seq_len in test_seq_lengths:
        print(f"\n  Testing seq_len={seq_len}...")
        
        # Generate test input
        batch_size, num_heads, head_dim = 2, 8, 128
        gamma = torch.rand(batch_size, num_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device=device) * 0.9 + 0.05
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = associative_scan_exponential(gamma)
                print(f"    ✓ Kernel executed successfully")
                
                # Check for fallback warnings
                if w:
                    for warning in w:
                        if "Triton" in str(warning.message) or "fallback" in str(warning.message):
                            print(f"    ⚠️ WARNING: {warning.message}")
                            return False
                
            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                return False
    
    print("\n  ✓ Triton kernel is working correctly (no fallback detected)")
    return True


def test_memory_complexity():
    """Test memory growth across sequence lengths to detect O(N²)."""
    print("\n" + "="*80)
    print("🔬 TEST 2: Memory Complexity Analysis (O(N²) Detection)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("  ⚠️ Memory profiling requires CUDA. Skipping.")
        return
    
    # Test sequence lengths (doubling pattern to detect O(N²))
    sequence_lengths = [16384, 32768, 65536]  # 16K, 32K, 64K
    
    # Create a small model for testing
    model = MMRecModel(
        vocab_size=10000,
        model_dim=1024,  # Smaller for testing
        num_layers=2,    # Fewer layers for faster testing
        num_heads=8,
        max_seq_len=65536
    ).to(device)
    
    print(f"\nModel: {model.get_num_params():,} parameters")
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    # Profile memory growth
    try:
        report = profile_memory_growth(
            model=model,
            sequence_lengths=sequence_lengths,
            batch_size=1,
            vocab_size=10000,
            device=device
        )
        
        # Check for O(N²) operations
        has_on2 = any(complexity == "O(N²)" for complexity in report.values())
        
        if has_on2:
            print("\n  ❌ CRITICAL: O(N²) memory growth detected!")
            print("   Operations with O(N²):")
            for op, complexity in report.items():
                if complexity == "O(N²)":
                    print(f"     - {op}: {complexity}")
            return False
        else:
            print("\n  ✓ No O(N²) memory growth detected")
            return True
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n  ❌ OOM at one of the test sequence lengths")
            print(f"   This indicates memory growth issue (possibly O(N²))")
            return False
        else:
            raise


def test_chunking():
    """Test chunking functionality for long sequences."""
    print("\n" + "="*80)
    print("🔬 TEST 3: Chunking Functionality")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with a long sequence (100K)
    seq_len = 100000
    chunk_size = 8192
    
    print(f"\n  Testing seq_len={seq_len} with chunk_size={chunk_size}")
    
    # Create a small model
    model = MMRecModel(
        vocab_size=10000,
        model_dim=1024,
        num_layers=2,
        num_heads=8,
        max_seq_len=seq_len
    ).to(device)
    
    # Generate input
    input_ids = torch.randint(0, 10000, (1, seq_len), device=device)
    
    try:
        # Test with chunking
        print("  Testing with chunking enabled...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        logits_chunked = model(input_ids, chunk_size=chunk_size)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"    ✓ Chunking successful")
        print(f"    Peak memory: {peak_memory_mb:.2f} MB")
        print(f"    Output shape: {logits_chunked.shape}")
        
        # Verify output is correct shape
        assert logits_chunked.shape == (1, seq_len, 10000), \
            f"Expected shape (1, {seq_len}, 10000), got {logits_chunked.shape}"
        
        print("  ✓ Chunking test PASSED")
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    ❌ OOM even with chunking!")
            print(f"    This indicates a deeper memory issue")
            return False
        else:
            raise
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False


def main():
    """Run all memory debugging tests."""
    print("\n" + "="*80)
    print("MM-Rec Memory Debugging Suite")
    print("="*80)
    print("\nThis script tests:")
    print("  1. Triton kernel fallback detection")
    print("  2. Memory complexity analysis (O(N²) detection)")
    print("  3. Chunking functionality for long sequences")
    
    results = {}
    
    # Test 1: Triton fallback
    results['triton'] = test_triton_fallback()
    
    # Test 2: Memory complexity
    results['memory'] = test_memory_complexity()
    
    # Test 3: Chunking
    results['chunking'] = test_chunking()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n  ✅ All tests PASSED - No critical memory issues detected")
    else:
        print("\n  ⚠️ Some tests FAILED - Critical memory issues detected")
        print("   Please review the warnings above and fix the issues.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

