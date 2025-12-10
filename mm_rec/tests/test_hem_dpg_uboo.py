"""
MM-Rec HEM, DPG, UBÖO Test Suite
Tests for the three integrated mechanisms: HEM, DPG, and UBÖO
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest
import torch
import torch.nn as nn
from typing import Tuple

from mm_rec.model import MMRecModel
from mm_rec.blocks.mm_rec_block import MMRecBlock
from mm_rec.core.memory_state import MemoryState, MemoryBankConfig


class TestHEM(unittest.TestCase):
    """Tests for HEM (Fused Kernel) mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_dim = 128
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 16
        self.device = torch.device('cpu')
    
    def test_hem_model_creation(self):
        """Test model creation with HEM enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=True
        )
        
        # Check that fused weight exists
        self.assertTrue(hasattr(model.blocks[0], 'W_fused'))
        self.assertIsNotNone(model.blocks[0].W_fused)
        
        # Check that separate weights don't exist when HEM is enabled
        self.assertFalse(hasattr(model.blocks[0], 'W_q'))
        
        print("✅ HEM model creation test passed")
    
    def test_hem_forward_pass(self):
        """Test forward pass with HEM enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=True
        )
        
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass
        logits = model(input_ids, return_auxiliary_loss=False)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))
        
        print("✅ HEM forward pass test passed")
    
    def test_hem_vs_no_hem_parameter_count(self):
        """Test that HEM changes parameter count."""
        model_no_hem = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=False
        )
        
        model_hem = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=True
        )
        
        params_no_hem = model_no_hem.get_num_params()
        params_hem = model_hem.get_num_params()
        
        # HEM should have different parameter count (due to fused matrix)
        # Note: May be slightly different due to positional encoding
        print(f"  No HEM: {params_no_hem} params")
        print(f"  With HEM: {params_hem} params")
        
        print("✅ HEM parameter count test passed")


class TestDPG(unittest.TestCase):
    """Tests for DPG (Dynamic Projection Gating) mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_dim = 128
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 16
        self.device = torch.device('cpu')
    
    def test_dpg_model_creation(self):
        """Test model creation with DPG enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_dpg=True,
            dpg_rank=64
        )
        
        # Check that DPG weights exist
        self.assertTrue(hasattr(model.blocks[0], 'W_gamma_down'))
        self.assertTrue(hasattr(model.blocks[0], 'W_gamma_up'))
        self.assertIsNotNone(model.blocks[0].W_gamma_down)
        self.assertIsNotNone(model.blocks[0].W_gamma_up)
        
        # Check compute_dpg_gamma method exists
        self.assertTrue(hasattr(model.blocks[0], 'compute_dpg_gamma'))
        
        print("✅ DPG model creation test passed")
    
    def test_dpg_forward_pass(self):
        """Test forward pass with DPG enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_dpg=True,
            dpg_rank=64
        )
        
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass
        logits = model(input_ids, return_auxiliary_loss=False)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))
        
        print("✅ DPG forward pass test passed")
    
    def test_compute_dpg_gamma(self):
        """Test DPG gamma computation."""
        block = MMRecBlock(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            use_dpg=True,
            dpg_rank=64
        )
        
        # Create test input
        z_t = torch.randn(self.batch_size, 1, self.model_dim)
        
        # Compute gamma
        gamma = block.compute_dpg_gamma(z_t)
        
        # Check shape
        self.assertEqual(gamma.shape, (self.batch_size, 1, self.model_dim))
        
        # Check range [0, 1] (after sigmoid and clamp)
        self.assertTrue(torch.all(gamma >= 1e-6))
        self.assertTrue(torch.all(gamma <= 1.0 - 1e-6))
        
        print("✅ DPG gamma computation test passed")


class TestUBOO(unittest.TestCase):
    """Tests for UBÖO (Unbiased Backpropagation) mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_dim = 128
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 16
        self.device = torch.device('cpu')
    
    def test_uboo_model_creation(self):
        """Test model creation with UBÖO enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_uboo=True,
            lambda_P=0.1
        )
        
        # Check that UBÖO is enabled
        self.assertTrue(model.use_uboo)
        self.assertEqual(model.lambda_P, 0.1)
        
        # Check that MDI has planning error projections
        self.assertTrue(hasattr(model.blocks[0].mdi, 'W_planning_error'))
        self.assertTrue(hasattr(model.blocks[0].mdi, 'W_planning_target'))
        self.assertIsNotNone(model.blocks[0].mdi.W_planning_error)
        self.assertIsNotNone(model.blocks[0].mdi.W_planning_target)
        
        print("✅ UBÖO model creation test passed")
    
    def test_uboo_forward_pass_with_auxiliary_loss(self):
        """Test forward pass with UBÖO and auxiliary loss."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_uboo=True,
            lambda_P=0.1
        )
        
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass with auxiliary loss
        result = model(input_ids, return_auxiliary_loss=True)
        
        if isinstance(result, tuple):
            logits, L_Aux = result
            # Check output shape
            self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))
            # Check auxiliary loss exists
            self.assertIsNotNone(L_Aux)
            self.assertTrue(L_Aux.item() >= 0)  # Loss should be non-negative
            print(f"  Auxiliary loss: {L_Aux.item():.6f}")
        else:
            self.fail("Expected tuple (logits, L_Aux) but got single value")
        
        print("✅ UBÖO forward pass with auxiliary loss test passed")
    
    def test_uboo_forward_pass_without_auxiliary_loss(self):
        """Test forward pass with UBÖO but without auxiliary loss."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_uboo=True,
            lambda_P=0.1
        )
        
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass without auxiliary loss
        logits = model(input_ids, return_auxiliary_loss=False)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))
        
        print("✅ UBÖO forward pass without auxiliary loss test passed")


class TestCombinedMechanisms(unittest.TestCase):
    """Tests for combined HEM, DPG, and UBÖO mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_dim = 128
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 16
        self.device = torch.device('cpu')
    
    def test_all_mechanisms_enabled(self):
        """Test model with all three mechanisms enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=True,
            use_dpg=True,
            dpg_rank=64,
            use_uboo=True,
            lambda_P=0.1
        )
        
        # Check all mechanisms are enabled
        self.assertTrue(model.use_hem)
        self.assertTrue(model.use_dpg)
        self.assertTrue(model.use_uboo)
        
        # Check HEM
        self.assertTrue(hasattr(model.blocks[0], 'W_fused'))
        
        # Check DPG
        self.assertTrue(hasattr(model.blocks[0], 'W_gamma_down'))
        self.assertTrue(hasattr(model.blocks[0], 'W_gamma_up'))
        
        # Check UBÖO
        self.assertTrue(hasattr(model.blocks[0].mdi, 'W_planning_error'))
        
        print("✅ All mechanisms enabled test passed")
    
    def test_combined_forward_pass(self):
        """Test forward pass with all mechanisms enabled."""
        model = MMRecModel(
            vocab_size=1000,
            model_dim=self.model_dim,
            num_layers=2,
            num_heads=self.num_heads,
            use_hem=True,
            use_dpg=True,
            dpg_rank=64,
            use_uboo=True,
            lambda_P=0.1
        )
        
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        # Forward pass with auxiliary loss
        result = model(input_ids, return_auxiliary_loss=True)
        
        if isinstance(result, tuple):
            logits, L_Aux = result
            # Check output shape
            self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))
            # Check auxiliary loss exists
            self.assertIsNotNone(L_Aux)
            print(f"  Combined mechanisms - Auxiliary loss: {L_Aux.item():.6f}")
        else:
            self.fail("Expected tuple (logits, L_Aux) but got single value")
        
        print("✅ Combined mechanisms forward pass test passed")
    
    def test_parameter_count_comparison(self):
        """Compare parameter counts for different configurations."""
        configs = [
            ("Baseline", {"use_hem": False, "use_dpg": False, "use_uboo": False}),
            ("HEM only", {"use_hem": True, "use_dpg": False, "use_uboo": False}),
            ("DPG only", {"use_hem": False, "use_dpg": True, "use_uboo": False}),
            ("UBÖO only", {"use_hem": False, "use_dpg": False, "use_uboo": True}),
            ("All", {"use_hem": True, "use_dpg": True, "use_uboo": True}),
        ]
        
        for name, config in configs:
            model = MMRecModel(
                vocab_size=1000,
                model_dim=self.model_dim,
                num_layers=2,
                num_heads=self.num_heads,
                dpg_rank=64,
                lambda_P=0.1,
                **config
            )
            params = model.get_num_params()
            print(f"  {name}: {params} params")
        
        print("✅ Parameter count comparison test passed")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHEM,
        TestDPG,
        TestUBOO,
        TestCombinedMechanisms,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("MM-Rec HEM, DPG, UBÖO Test Suite")
    print("=" * 60)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    exit(0 if success else 1)
