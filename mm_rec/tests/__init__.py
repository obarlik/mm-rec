"""
MM-Rec Test Suite
Organized test modules for MM-Rec architecture
"""

__version__ = "0.1.0"

# Test categories
__all__ = [
    # Core component tests
    "test_components",
    # Associative scan tests
    "test_associative_scan_validation",
    # Long sequence tests
    "test_32k_sequence",
    # Gradient tests
    "test_gradients",
    "test_gradient_flow_detailed",
]
