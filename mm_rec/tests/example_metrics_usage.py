"""
Example: How to use metrics collection in tests

This file demonstrates how to use the metrics collection infrastructure
without polluting test code.
"""

import torch
from mm_rec.tests.test_metrics import record_test_metric, record_memory_stats, save_test_metrics


def example_test_with_metrics():
    """Example test showing how to record metrics."""
    
    # Metrics are automatically collected via conftest.py fixture
    # But you can also manually record specific metrics:
    
    # Record model parameters
    model = torch.nn.Linear(100, 50)
    num_params = sum(p.numel() for p in model.parameters())
    record_test_metric("num_parameters", num_params)
    
    # Record sequence length
    seq_len = 1024
    record_test_metric("sequence_length", seq_len)
    
    # Record batch size
    batch_size = 8
    record_test_metric("batch_size", batch_size)
    
    # Record chunk size (if using chunking)
    chunk_size = 256
    record_test_metric("chunk_size", chunk_size)
    
    # Record memory stats at any point
    record_memory_stats()
    
    # Do some computation
    x = torch.randn(batch_size, seq_len, 100)
    y = model(x)
    
    # Record throughput if applicable
    import time
    start = time.time()
    # ... computation ...
    elapsed = time.time() - start
    throughput = (batch_size * seq_len) / elapsed
    record_test_metric("throughput_tokens_per_sec", throughput)
    
    # Record final memory
    record_memory_stats()
    
    # Metrics are automatically saved at the end of the test
    # But you can also manually save:
    # save_test_metrics("custom_metrics.json")


# Example: Using metrics in a real test
def test_model_forward_with_metrics():
    """Example test that records metrics."""
    from mm_rec.models.mmrec_100m import MMRec100M
    
    # Create model
    model = MMRec100M(
        vocab_size=1000,
        expert_dim=256,
        num_layers=2,
        num_heads=4,
        ffn_dim=512
    )
    
    # Record model info
    num_params = sum(p.numel() for p in model.parameters())
    record_test_metric("num_parameters", num_params)
    
    # Record test configuration
    batch_size = 2
    seq_len = 512
    record_test_metric("batch_size", batch_size)
    record_test_metric("sequence_length", seq_len)
    
    # Record memory before forward pass
    record_memory_stats()
    
    # Forward pass
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    
    # Record memory after forward pass
    record_memory_stats()
    
    # Assertions
    assert logits.shape == (batch_size, seq_len, 1000)
    
    # Metrics are automatically saved by conftest.py fixture

