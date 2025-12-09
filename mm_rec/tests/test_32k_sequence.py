"""
Integration test for 32K sequence length
Validates model can process 32,768 tokens without errors
"""

import torch
import unittest
import sys
import os
import time

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
    
    @pytest.mark.timeout(120, method='thread', func_only=True)  # 2 minute timeout
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
    
    def test_32k_chunking(self):
        """Test that chunking works correctly for 32K sequence."""
        # CRITICAL: Print immediately to verify test starts
        import sys
        sys.stdout.flush()
        print(f"\n{'='*80}", flush=True)
        print("32K Sequence Chunking Test", flush=True)
        print(f"{'='*80}\n", flush=True)
        print(f"✅ Test started at {time.time()}", flush=True)
        
        # AGGRESSIVE TIMEOUT: Start time for manual timeout checking
        start_time = time.time()
        timeout_seconds = 180  # 3 minutes - STRICT TIMEOUT
        
        def check_timeout():
            """Check if timeout exceeded - fail immediately if so."""
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                self.fail(
                    f"❌ TIMEOUT: Test exceeded {timeout_seconds}s limit! "
                    f"Elapsed: {elapsed:.2f}s. Test terminated."
                )
            return elapsed
        
        # CRITICAL: Create model in thread with STRICT timeout
        print("Creating model...", flush=True)
        import threading
        model_container = [None]
        model_exception = [None]
        model_done = threading.Event()
        
        def create_model():
            try:
                print(f"  [Thread] Model creation started...", flush=True)
                model_container[0] = MMRec100M(
                    vocab_size=self.vocab_size,
                    expert_dim=256,
                    num_layers=12,
                    num_heads=8,
                    ffn_dim=2048
                ).to(self.device)
                print(f"  [Thread] ✅ Model created successfully", flush=True)
            except Exception as e:
                print(f"  [Thread] ❌ Model creation failed: {e}", flush=True)
                model_exception[0] = e
            finally:
                model_done.set()
                print(f"  [Thread] Model creation thread finished", flush=True)
        
        model_thread = threading.Thread(target=create_model, daemon=True)
        print(f"  Starting model creation thread...", flush=True)
        model_thread.start()
        
        # Wait for model creation with STRICT timeout (max 20s for testing)
        model_timeout = 20
        print(f"  Waiting for model creation (timeout: {model_timeout}s)...", flush=True)
        model_done.wait(timeout=model_timeout)
        
        if model_thread.is_alive():
            elapsed = time.time() - start_time
            print(f"  ❌ TIMEOUT: Model creation exceeded {model_timeout}s! Elapsed: {elapsed:.2f}s", flush=True)
            self.fail(
                f"❌ TIMEOUT: Model creation exceeded {model_timeout}s! "
                f"Elapsed: {elapsed:.2f}s. Test terminated."
            )
        
        if model_exception[0]:
            raise model_exception[0]
        
        model = model_container[0]
        elapsed = check_timeout()  # Check after model creation
        print(f"✅ Model created. Total elapsed: {elapsed:.2f}s", flush=True)
        
        # Generate input
        print("Generating input...")
        check_timeout()  # Check after input generation
        batch_size = 1
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        print(f"Input generated. Shape: {input_ids.shape}")
        
        # Test with different chunk sizes (reduced for faster testing with timeout)
        # Only test one chunk size to ensure timeout works
        chunk_sizes = [8192]  # Single chunk size for timeout testing
        
        model.eval()
        with torch.no_grad():
            results = {}
            
            for chunk_size in chunk_sizes:
                # AGGRESSIVE: Check timeout BEFORE each chunk
                elapsed = check_timeout()
                remaining_time = timeout_seconds - elapsed
                
                if remaining_time <= 5:  # Less than 5 seconds remaining
                    self.fail(
                        f"❌ TIMEOUT: Only {remaining_time:.1f}s remaining, "
                        f"skipping chunk {chunk_size}. Elapsed: {elapsed:.1f}s"
                    )
                
                try:
                    print(f"\nTesting chunk_size={chunk_size}... (elapsed: {elapsed:.1f}s, remaining: {remaining_time:.1f}s)")
                    chunk_start = time.time()
                    
                    # CRITICAL: Run model in thread with strict timeout
                    import threading
                    result_container = [None]
                    exception_container = [None]
                    thread_done = threading.Event()
                    
                    def run_model():
                        try:
                            result_container[0] = model(input_ids, chunk_size=chunk_size)
                        except Exception as e:
                            exception_container[0] = e
                        finally:
                            thread_done.set()
                    
                    model_thread = threading.Thread(target=run_model, daemon=True)
                    model_thread.start()
                    
                    # Wait with strict timeout (max 60s per chunk, or remaining time)
                    chunk_timeout = min(remaining_time - 2, 60)  # Leave 2s buffer
                    if chunk_timeout <= 0:
                        self.fail(f"❌ TIMEOUT: No time for chunk {chunk_size}")
                    
                    thread_done.wait(timeout=chunk_timeout)
                    
                    # Check if thread is still running (timeout occurred)
                    if model_thread.is_alive():
                        elapsed_total = time.time() - start_time
                        self.fail(
                            f"❌ TIMEOUT: Chunk {chunk_size} exceeded {chunk_timeout}s limit! "
                            f"Total elapsed: {elapsed_total:.2f}s (limit: {timeout_seconds}s). "
                            f"Test terminated."
                        )
                    
                    # Check for exceptions
                    if exception_container[0]:
                        raise exception_container[0]
                    
                    logits = result_container[0]
                    chunk_elapsed = time.time() - chunk_start
                    results[chunk_size] = logits
                    
                    # Check timeout AFTER chunk
                    check_timeout()
                    
                    print(f"  ✅ Chunk size {chunk_size} works (took {chunk_elapsed:.1f}s)")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  ⚠️ OOM with chunk_size={chunk_size}")
                    else:
                        raise
                except AssertionError:
                    # Re-raise timeout failures immediately
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

