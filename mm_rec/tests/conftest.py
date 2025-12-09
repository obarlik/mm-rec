"""
Pytest configuration and fixtures for MM-Rec tests
"""

import pytest
import torch
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import metrics collector
try:
    from mm_rec.tests.test_metrics import get_metrics_collector, record_test_metric, record_memory_stats
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for metrics."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{call.when}", rep)
    return rep


@pytest.fixture(scope="function", autouse=True)
def test_metrics(request):
    """Automatic fixture to collect metrics for each test."""
    if not METRICS_AVAILABLE:
        yield
        return
    
    collector = get_metrics_collector()
    
    # Get test name and file
    if hasattr(request.node, 'parent') and request.node.parent:
        test_name = f"{request.node.parent.name}::{request.node.name}"
    else:
        test_name = request.node.name
    
    test_file = os.path.basename(str(request.node.fspath))
    
    # Start test tracking
    collector.start_test(test_name, test_file)
    collector.record_metric("test_file", test_file, test_name)
    
    # Record initial memory
    record_memory_stats()
    
    # Yield to test
    yield
    
    # Record final memory
    record_memory_stats()
    
    # End test tracking - get status from hook
    status = "passed"
    error_message = None
    
    if hasattr(request.node, 'rep_call'):
        if request.node.rep_call.failed:
            status = "failed"
            if hasattr(request.node.rep_call, 'longrepr'):
                error_message = str(request.node.rep_call.longrepr)
        elif request.node.rep_call.skipped:
            status = "skipped"
            if hasattr(request.node.rep_call, 'longrepr'):
                error_message = str(request.node.rep_call.longrepr)
    
    collector.end_test(test_name, status, error_message)


# Pytest markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "long: marks tests that require long sequences (32K+)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "cpu: marks tests that run on CPU"
    )
    config.addinivalue_line(
        "markers", "extension: marks tests that require C++ extensions"
    )


def pytest_sessionfinish(session, exitstatus):
    """Save metrics at the end of test session."""
    if not METRICS_AVAILABLE:
        return
    
    collector = get_metrics_collector()
    
    # Save metrics
    metrics_file = collector.save_metrics()
    
    # Print summary
    summary = collector.get_summary()
    if summary:
        print("\n" + "=" * 80)
        print("Test Metrics Summary")
        print("=" * 80)
        print(f"Total tests: {summary['total_tests']}")
        if 'status_counts' in summary:
            print(f"Status counts: {summary['status_counts']}")
        if 'duration_stats' in summary:
            dur = summary['duration_stats']
            print(f"Duration: total={dur['total']:.2f}s, mean={dur['mean']:.2f}s, "
                  f"min={dur['min']:.2f}s, max={dur['max']:.2f}s")
        if 'gpu_memory_stats' in summary:
            mem = summary['gpu_memory_stats']
            print(f"GPU Memory: mean={mem['mean_mb']:.2f}MB, "
                  f"max={mem['max_mb']:.2f}MB, min={mem['min_mb']:.2f}MB")
        print(f"\nMetrics saved to: {metrics_file}")
        print("=" * 80)


@pytest.fixture(scope="session")
def device():
    """Get device for tests (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    """Get dtype for tests (BF16 if available, else FP32)."""
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


@pytest.fixture(scope="session")
def small_config():
    """Small test configuration for fast tests."""
    return {
        'vocab_size': 100,
        'model_dim': 64,
        'num_layers': 1,
        'num_heads': 4,
        'num_memories': 1,
        'mem_dim': 32,
        'max_seq_len': 128,
        'seq_len': 32,
        'batch_size': 2
    }


@pytest.fixture(scope="session")
def medium_config():
    """Medium test configuration."""
    return {
        'vocab_size': 1000,
        'model_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'num_memories': 1,
        'mem_dim': 64,
        'max_seq_len': 512,
        'seq_len': 256,
        'batch_size': 2
    }

