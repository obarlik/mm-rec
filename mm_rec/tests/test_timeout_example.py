"""
Example test with timeout decorator
Shows how to add timeouts to individual tests
"""

import unittest
import pytest
import time
import torch


class TestTimeoutExample(unittest.TestCase):
    """Example tests showing timeout usage."""
    
    @pytest.mark.timeout(5)  # 5 second timeout
    def test_fast_operation(self):
        """Fast test that should complete quickly."""
        x = torch.rand(10, 10)
        y = x @ x
        self.assertIsNotNone(y)
    
    @pytest.mark.timeout(10)  # 10 second timeout
    def test_medium_operation(self):
        """Medium test with longer timeout."""
        x = torch.rand(100, 100)
        y = x @ x
        self.assertIsNotNone(y)
    
    @pytest.mark.timeout(30)  # 30 second timeout
    @pytest.mark.slow
    def test_slow_operation(self):
        """Slow test with extended timeout."""
        # Simulate slow operation
        time.sleep(1)
        x = torch.rand(1000, 1000)
        y = x @ x
        self.assertIsNotNone(y)
    
    def test_no_timeout_specified(self):
        """Test without explicit timeout uses pytest.ini default."""
        x = torch.rand(10, 10)
        y = x @ x
        self.assertIsNotNone(y)

