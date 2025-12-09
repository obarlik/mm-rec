"""
Unit test for Associative Scan kernel validation
Compares Triton kernel output with sequential Python implementation
"""

import torch
import unittest
import sys
import os
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.core.associative_scan_triton import associative_scan_exponential
from mm_rec.core.associative_scan_hybrid import associative_scan_exponential_hybrid


def sequential_cumulative_product(gamma: torch.Tensor) -> torch.Tensor:
    """
    Sequential reference implementation for cumulative exponential product.
    
    Args:
        gamma: [batch, heads, seq_len, head_dim]
    
    Returns:
        cumulative_product: [batch, heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = gamma.shape
    
    # Convert to FP32 for stability
    gamma_fp32 = gamma.to(torch.float32)
    
    # Sequential cumulative product
    cumulative = torch.zeros_like(gamma_fp32)
    for t in range(seq_len):
        if t == 0:
            cumulative[:, :, t, :] = gamma_fp32[:, :, t, :]
        else:
            cumulative[:, :, t, :] = cumulative[:, :, t-1, :] * gamma_fp32[:, :, t, :]
    
    return cumulative.to(gamma.dtype)


class TestAssociativeScanValidation(unittest.TestCase):
    """Test associative scan kernel correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rtol = 1e-3
        self.atol = 1e-4
    
    def test_short_sequence(self):
        """Test with short sequence (128 tokens)."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        
        # Generate test input
        gamma = torch.rand(batch_size, num_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device=self.device) * 0.9 + 0.05
        
        # Reference (sequential)
        ref_result = sequential_cumulative_product(gamma)
        
        # Triton kernel
        try:
            triton_result = associative_scan_exponential(gamma)
        except Exception as e:
            self.skipTest(f"Triton kernel not available: {e}")
        
        # Compare
        max_diff = torch.max(torch.abs(triton_result - ref_result)).item()
        mean_diff = torch.mean(torch.abs(triton_result - ref_result)).item()
        
        print(f"\nShort sequence test:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        self.assertTrue(
            torch.allclose(triton_result, ref_result, rtol=self.rtol, atol=self.atol),
            f"Triton result does not match reference. Max diff: {max_diff:.6e}"
        )
    
    @pytest.mark.slow
    @pytest.mark.timeout(10, method='thread', func_only=True)  # 10 second timeout for 1024 tokens
    def test_medium_sequence(self):
        """Test with medium sequence (1024 tokens)."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
        
        # Generate test input
        gamma = torch.rand(batch_size, num_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device=self.device) * 0.9 + 0.05
        
        # Reference (sequential)
        ref_result = sequential_cumulative_product(gamma)
        
        # Triton kernel
        try:
            triton_result = associative_scan_exponential(gamma)
        except Exception as e:
            self.skipTest(f"Triton kernel not available: {e}")
        
        # Compare
        max_diff = torch.max(torch.abs(triton_result - ref_result)).item()
        
        print(f"\nMedium sequence test:")
        print(f"  Max difference: {max_diff:.6e}")
        
        self.assertTrue(
            torch.allclose(triton_result, ref_result, rtol=self.rtol, atol=self.atol),
            f"Triton result does not match reference. Max diff: {max_diff:.6e}"
        )
    
    @pytest.mark.long
    @pytest.mark.slow
    @pytest.mark.timeout(30, method='thread', func_only=True)  # 30 second timeout for 8192 tokens
    def test_long_sequence(self):
        """Test with long sequence (8192 tokens)."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 8192, 32
        
        # Generate test input
        gamma = torch.rand(batch_size, num_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device=self.device) * 0.9 + 0.05
        
        # Reference (sequential) - may be slow
        print("\nComputing reference (sequential)...")
        ref_result = sequential_cumulative_product(gamma)
        
        # Triton kernel
        try:
            print("Computing with Triton kernel...")
            triton_result = associative_scan_exponential(gamma)
        except Exception as e:
            self.skipTest(f"Triton kernel not available: {e}")
        
        # Compare
        max_diff = torch.max(torch.abs(triton_result - ref_result)).item()
        
        print(f"\nLong sequence test:")
        print(f"  Max difference: {max_diff:.6e}")
        
        self.assertTrue(
            torch.allclose(triton_result, ref_result, rtol=self.rtol, atol=self.atol),
            f"Triton result does not match reference. Max diff: {max_diff:.6e}"
        )
    
    def test_hybrid_precision(self):
        """Test hybrid precision (BF16 + FP64 log accumulation)."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
        
        # Generate test input (BF16)
        gamma = torch.rand(batch_size, num_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device=self.device) * 0.9 + 0.05
        
        # Reference (sequential)
        ref_result = sequential_cumulative_product(gamma)
        
        # Hybrid precision kernel
        try:
            hybrid_result = associative_scan_exponential_hybrid(gamma)
        except Exception as e:
            self.skipTest(f"Hybrid kernel not available: {e}")
        
        # Compare
        max_diff = torch.max(torch.abs(hybrid_result - ref_result)).item()
        
        print(f"\nHybrid precision test:")
        print(f"  Max difference: {max_diff:.6e}")
        
        self.assertTrue(
            torch.allclose(hybrid_result, ref_result, rtol=self.rtol, atol=self.atol),
            f"Hybrid result does not match reference. Max diff: {max_diff:.6e}"
        )
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
        
        # Test with very small values (risk of underflow)
        gamma_small = torch.ones(batch_size, num_heads, seq_len, head_dim,
                                 dtype=torch.bfloat16, device=self.device) * 0.001
        
        # Test with values close to 1 (risk of overflow in product)
        gamma_large = torch.ones(batch_size, num_heads, seq_len, head_dim,
                                dtype=torch.bfloat16, device=self.device) * 0.999
        
        for name, gamma in [("small", gamma_small), ("large", gamma_large)]:
            try:
                result = associative_scan_exponential(gamma)
                
                # Check for NaN/Inf
                self.assertFalse(
                    torch.isnan(result).any(),
                    f"NaN detected in {name} value test"
                )
                self.assertFalse(
                    torch.isinf(result).any(),
                    f"Inf detected in {name} value test"
                )
                
                # Check that result is reasonable
                self.assertTrue(
                    (result >= 0).all() and (result <= 1).all(),
                    f"Result out of bounds in {name} value test"
                )
                
                print(f"\n✅ {name} value test passed")
            except Exception as e:
                self.fail(f"{name} value test failed: {e}")


if __name__ == '__main__':
    unittest.main()

