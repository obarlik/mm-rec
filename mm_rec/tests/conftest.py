"""
Pytest configuration and fixtures for MM-Rec tests
"""

import pytest
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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

