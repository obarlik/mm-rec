"""
Detailed Gradient Flow Analysis
Tests to identify which parameters don't receive gradients and why
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.core.memory_state import MemoryState
from mm_rec.tests.test_gradients import assert_all_parameters_receive_gradients


class TestGradientFlowDetailed(unittest.TestCase):
    """Detailed tests to identify gradient flow issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        
        self.config = {
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
    
    def create_model_and_inputs(self):
        """Create model and input tensors."""
        model = MMRecModel(
            vocab_size=self.config['vocab_size'],
            model_dim=self.config['model_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            num_memories=self.config['num_memories'],
            mem_dim=self.config['mem_dim'],
            max_seq_len=self.config['max_seq_len'],
            dropout=0.0
        )
        model = model.to(self.device)
        model.train()
        
        input_ids = torch.randint(
            0, self.config['vocab_size'],
            size=(self.config['batch_size'], self.config['seq_len']),
            device=self.device,
            dtype=torch.long
        )
        
        return model, input_ids
    
    def test_all_parameters_receive_gradients(self):
        """
        Test that ALL parameters receive gradients after forward/backward pass.
        
        This test will fail if any parameter doesn't receive gradients,
        helping identify which components need fixing.
        """
        model, input_ids = self.create_model_and_inputs()
        
        # Forward pass
        logits = model(input_ids)
        
        # Create targets
        targets = torch.randint(
            0, self.config['vocab_size'],
            size=(self.config['batch_size'], self.config['seq_len']),
            device=self.device,
            dtype=torch.long
        )
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = loss_fn(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Assert all parameters receive gradients
        # This will raise AssertionError with detailed information if any parameter lacks gradients
        assert_all_parameters_receive_gradients(model)
        
        print("✓ All parameters receive gradients")
    
    def test_identify_parameters_without_gradients(self):
        """
        Identify which specific parameters don't receive gradients.
        
        This test provides detailed output for debugging.
        """
        model, input_ids = self.create_model_and_inputs()
        
        # Forward pass
        logits = model(input_ids)
        
        # Create targets and compute loss
        targets = torch.randint(
            0, self.config['vocab_size'],
            size=(self.config['batch_size'], self.config['seq_len']),
            device=self.device,
            dtype=torch.long
        )
        
        loss_fn = nn.CrossEntropyLoss()
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = loss_fn(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradient flow
        params_with_grad = []
        params_without_grad = []
        params_with_nan = []
        params_with_inf = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_without_grad.append(name)
                elif torch.isnan(param.grad).any():
                    params_with_nan.append(name)
                elif torch.isinf(param.grad).any():
                    params_with_inf.append(name)
                else:
                    params_with_grad.append(name)
        
        # Print detailed report
        print(f"\n{'='*60}")
        print("Gradient Flow Analysis")
        print(f"{'='*60}")
        print(f"Parameters with gradients: {len(params_with_grad)}")
        print(f"Parameters without gradients: {len(params_without_grad)}")
        print(f"Parameters with NaN gradients: {len(params_with_nan)}")
        print(f"Parameters with Inf gradients: {len(params_with_inf)}")
        
        if params_without_grad:
            print(f"\n⚠ Parameters WITHOUT gradients ({len(params_without_grad)}):")
            for name in params_without_grad:
                param = dict(model.named_parameters())[name]
                print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        
        if params_with_nan:
            print(f"\n⚠ Parameters with NaN gradients ({len(params_with_nan)}):")
            for name in params_with_nan:
                print(f"  - {name}")
        
        if params_with_inf:
            print(f"\n⚠ Parameters with Inf gradients ({len(params_with_inf)}):")
            for name in params_with_inf:
                print(f"  - {name}")
        
        # Group by component
        component_groups = {}
        for name in params_without_grad:
            component = name.split('.')[0] if '.' in name else name
            if component not in component_groups:
                component_groups[component] = []
            component_groups[component].append(name)
        
        if component_groups:
            print(f"\n⚠ Grouped by component:")
            for component, params in component_groups.items():
                print(f"  {component}: {len(params)} parameters")
                for param_name in params[:3]:  # Show first 3
                    print(f"    - {param_name}")
                if len(params) > 3:
                    print(f"    ... and {len(params) - 3} more")
        
        print(f"{'='*60}\n")
        
        # Don't fail - just report
        # Uncomment to make this a hard failure:
        # if params_without_grad:
        #     self.fail(f"Found {len(params_without_grad)} parameters without gradients")


if __name__ == '__main__':
    unittest.main()

