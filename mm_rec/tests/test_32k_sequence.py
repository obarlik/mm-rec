"""
Integration test for 32K sequence length
Validates model can process 32,768 tokens without errors
"""

import torch
import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M


import pytest


@pytest.mark.long
@pytest.mark.slow
class Test32KSequence(unittest.TestCase):
    """Test 32K sequence length processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = 32000
        self.seq_len = 32768  # 32K tokens
    
    @pytest.mark.timeout(120, method='thread')  # 2 minute timeout for 32K test
    def test_32k_forward_pass(self):
        """Test forward pass with 32K sequence."""
        print(f"\n{'='*80}")
        print("32K Sequence Length Test")
        print(f"{'='*80}\n")
        
        # Create model
        model = MMRec100M(
            vocab_size=self.vocab_size,
            expert_dim=256,
            num_layers=12,
            num_heads=8,
            ffn_dim=2048
        ).to(self.device)
        
        print(f"Model parameters: {model.get_num_params():,}")
        print(f"Device: {self.device}")
        print(f"Sequence length: {self.seq_len:,} tokens")
        
        # Generate input
        batch_size = 1
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass with chunking (automatic for seq_len > 32K)
        model.eval()
        with torch.no_grad():
            try:
                print("\nRunning forward pass...")
                logits = model(input_ids, chunk_size=8192)  # Use 8K chunks
                
                print(f"✅ Forward pass completed!")
                print(f"Output shape: {logits.shape}")
                print(f"Expected: [{batch_size}, {self.seq_len}, {self.vocab_size}]")
                
                # Validate output shape
                self.assertEqual(logits.shape, (batch_size, self.seq_len, self.vocab_size))
                
                # Check for NaN/Inf
                self.assertFalse(
                    torch.isnan(logits).any(),
                    "NaN detected in output"
                )
                self.assertFalse(
                    torch.isinf(logits).any(),
                    "Inf detected in output"
                )
                
                # Check output is reasonable
                self.assertTrue(
                    torch.isfinite(logits).all(),
                    "Non-finite values in output"
                )
                
                print("\n✅ All checks passed!")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.skipTest(f"OOM on device: {e}")
                else:
                    raise
    
    @pytest.mark.timeout(120, method='thread')  # 2 minute timeout
    def test_32k_with_memory_states(self):
        """Test 32K sequence with explicit memory states."""
        print(f"\n{'='*80}")
        print("32K Sequence with Memory States Test")
        print(f"{'='*80}\n")
        
        # Create model
        model = MMRec100M(
            vocab_size=self.vocab_size,
            expert_dim=256,
            num_layers=12,
            num_heads=8,
            ffn_dim=2048
        ).to(self.device)
        
        # Create memory states
        from mm_rec.core.memory_state import MemoryState
        
        batch_size = 1
        memory_states = {
            "text": [
                model.text_expert._create_memory_state(batch_size, self.device)
                for _ in range(model.num_layers)
            ],
            "code": [
                model.code_expert._create_memory_state(batch_size, self.device)
                for _ in range(model.num_layers)
            ]
        }
        
        # Generate input
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                logits, updated_states = model(
                    input_ids,
                    memory_states=memory_states,
                    chunk_size=8192,
                    return_memory=True
                )
                
                print(f"✅ Forward pass with memory states completed!")
                print(f"Output shape: {logits.shape}")
                
                # Validate memory states were updated
                self.assertIsNotNone(updated_states)
                self.assertIn("text", updated_states)
                self.assertIn("code", updated_states)
                
                print("\n✅ Memory state test passed!")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.skipTest(f"OOM on device: {e}")
                else:
                    raise
    
    @pytest.mark.timeout(180, method='thread')  # 3 minute timeout (multiple chunk sizes)
    def test_32k_chunking(self):
        """Test that chunking works correctly for 32K sequence."""
        print(f"\n{'='*80}")
        print("32K Sequence Chunking Test")
        print(f"{'='*80}\n")
        
        # Create model
        model = MMRec100M(
            vocab_size=self.vocab_size,
            expert_dim=256,
            num_layers=12,
            num_heads=8,
            ffn_dim=2048
        ).to(self.device)
        
        # Generate input
        batch_size = 1
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        # Test with different chunk sizes
        chunk_sizes = [4096, 8192, 16384]
        
        model.eval()
        with torch.no_grad():
            results = {}
            
            for chunk_size in chunk_sizes:
                try:
                    print(f"\nTesting chunk_size={chunk_size}...")
                    logits = model(input_ids, chunk_size=chunk_size)
                    results[chunk_size] = logits
                    print(f"  ✅ Chunk size {chunk_size} works")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  ⚠️ OOM with chunk_size={chunk_size}")
                    else:
                        raise
            
            # Compare results (should be similar)
            if len(results) >= 2:
                chunk_sizes_list = list(results.keys())
                logits1 = results[chunk_sizes_list[0]]
                logits2 = results[chunk_sizes_list[1]]
                
                max_diff = torch.max(torch.abs(logits1 - logits2)).item()
                print(f"\nMax difference between chunk sizes: {max_diff:.6e}")
                
                # Results should be close (within numerical precision)
                self.assertTrue(
                    torch.allclose(logits1, logits2, rtol=1e-3, atol=1e-4),
                    f"Results differ significantly between chunk sizes. Max diff: {max_diff:.6e}"
                )
                
                print("\n✅ Chunking consistency test passed!")


if __name__ == '__main__':
    unittest.main()

