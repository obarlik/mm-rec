"""
MM-Rec Component Tests
Unit tests for core components functionality
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict

# Import MM-Rec components
from mm_rec.core.memory_state import MemoryBank, MemoryState, MemoryBankConfig
from mm_rec.core.mdi import MemoryDecayIntegration
from mm_rec.core.hds import HierarchicalDataStructure
from mm_rec.blocks.attention import MultiMemoryAttention
from mm_rec.blocks.mm_rec_block import MMRecBlock


# Test configuration
TEST_CONFIG: Dict = {
    'model_dim': 256,
    'inner_dim': 128,
    'num_heads': 8,
    'seq_len': 64,
    'batch_size': 2,
    'num_memories': 1,
    'mem_dim': 128,
    'vocab_size': 1000,
    'dtype': torch.float32
}


class TestMemoryStateManagement(unittest.TestCase):
    """Tests for Memory State Management components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
    
    def test_memory_bank_initialization(self):
        """Test MemoryBank initialization and tensor dimensions."""
        k_dim = self.config['model_dim']
        v_dim = self.config['model_dim']
        num_slots = self.config['seq_len']
        
        bank = MemoryBank(
            k_dim=k_dim,
            v_dim=v_dim,
            num_slots=num_slots,
            dtype=self.config['dtype'],
            device=self.device
        )
        
        # Check that k and v are initialized
        self.assertIsNotNone(bank.k)
        self.assertIsNotNone(bank.v)
        
        # Check tensor dimensions
        self.assertEqual(bank.k.shape, (num_slots, k_dim))
        self.assertEqual(bank.v.shape, (num_slots, v_dim))
        
        # Check data type
        self.assertEqual(bank.k.dtype, self.config['dtype'])
        self.assertEqual(bank.v.dtype, self.config['dtype'])
        
        # Check device
        self.assertEqual(bank.k.device, self.device)
        self.assertEqual(bank.v.device, self.device)
    
    def test_memory_state_initialization(self):
        """Test MemoryState initialization and short/long-term memory."""
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,  # M << seq_len
            'dtype': self.config['dtype']
        }
        
        state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
        
        # Check that banks are initialized
        self.assertIsNotNone(state.short_term)
        self.assertIsNotNone(state.long_term)
        
        # Check short-term memory dimensions
        k_short, v_short = state.get_state('short')
        self.assertEqual(k_short.shape[0], self.config['seq_len'])
        self.assertEqual(v_short.shape[0], self.config['seq_len'])
        
        # Check long-term memory dimensions
        k_long, v_long = state.get_state('long')
        self.assertEqual(k_long.shape[0], 1024)  # M slots
        self.assertEqual(v_long.shape[0], 1024)
        
        # Check device
        self.assertEqual(k_short.device, self.device)
        self.assertEqual(k_long.device, self.device)
    
    def test_memory_state_update(self):
        """Test MemoryState update functionality."""
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,
            'dtype': self.config['dtype']
        }
        
        state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
        
        # Create new tensors for update
        new_k = torch.randn(self.config['seq_len'], self.config['model_dim'], 
                          dtype=self.config['dtype'], device=self.device)
        new_v = torch.randn(self.config['seq_len'], self.config['model_dim'],
                          dtype=self.config['dtype'], device=self.device)
        
        # Update short-term memory
        state.update_state('short', new_k, new_v)
        
        # Verify update
        k_updated, v_updated = state.get_state('short')
        torch.testing.assert_close(k_updated, new_k)
        torch.testing.assert_close(v_updated, new_v)


class TestMDI(unittest.TestCase):
    """Tests for Memory Decay/Integration component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
    
    def test_mdi_forward_pass(self):
        """Test MemoryDecayIntegration forward pass."""
        mdi = MemoryDecayIntegration(
            model_dim=self.config['model_dim'],
            inner_dim=self.config['inner_dim'],
            use_context_modulation=True
        )
        
        # Create random inputs
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        z_t = torch.randn(batch_size, seq_len, model_dim,
                         dtype=self.config['dtype'], device=self.device)
        h_prev = torch.randn(batch_size, seq_len, model_dim,
                            dtype=self.config['dtype'], device=self.device)
        context = torch.randn(batch_size, seq_len, model_dim,
                             dtype=self.config['dtype'], device=self.device)
        
        # Forward pass
        h_new, gamma = mdi(z_t, h_prev, context=context)
        
        # Check output dimensions
        self.assertEqual(h_new.shape, (batch_size, seq_len, model_dim))
        self.assertEqual(gamma.shape, (batch_size, seq_len, model_dim))
        
        # Check that gamma is in valid range [1e-6, 1-1e-6]
        self.assertTrue(torch.all(gamma >= 1e-6))
        self.assertTrue(torch.all(gamma <= 1.0 - 1e-6))
        
        # Check that outputs are not NaN or Inf
        self.assertFalse(torch.isnan(h_new).any())
        self.assertFalse(torch.isinf(h_new).any())
        self.assertFalse(torch.isnan(gamma).any())
        self.assertFalse(torch.isinf(gamma).any())
    
    def test_mdi_decay_only(self):
        """Test MDI decay coefficient computation only."""
        mdi = MemoryDecayIntegration(
            model_dim=self.config['model_dim'],
            inner_dim=self.config['inner_dim']
        )
        
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        z_t = torch.randn(batch_size, seq_len, model_dim,
                         dtype=self.config['dtype'], device=self.device)
        
        gamma = mdi.compute_decay_only(z_t)
        
        # Check dimensions
        self.assertEqual(gamma.shape, (batch_size, seq_len, model_dim))
        
        # Check valid range
        self.assertTrue(torch.all(gamma >= 1e-6))
        self.assertTrue(torch.all(gamma <= 1.0 - 1e-6))


class TestHDS(unittest.TestCase):
    """Tests for Hierarchical Data Structure component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
        
        # Create memory state for HDS
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,
            'dtype': self.config['dtype']
        }
        
        self.memory_state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
    
    def test_hds_initialization(self):
        """Test HDS initialization."""
        hds = HierarchicalDataStructure(
            memory_state=self.memory_state,
            num_levels=3,
            model_dim=self.config['model_dim']
        )
        
        self.assertIsNotNone(hds)
        self.assertEqual(hds.num_levels, 3)
        self.assertEqual(hds.model_dim, self.config['model_dim'])
    
    def test_hds_hierarchy_construction(self):
        """Test HDS hierarchy construction."""
        hds = HierarchicalDataStructure(
            memory_state=self.memory_state,
            num_levels=3,
            model_dim=self.config['model_dim']
        )
        
        # Construct hierarchy
        hds.construct_hierarchy()
        
        # Check that hierarchy is constructed
        self.assertTrue(hds._hierarchy_constructed)
        self.assertGreater(len(hds.levels_cache), 0)
        
        # Check that all levels exist
        for level in range(hds.num_levels):
            self.assertIn(level, hds.levels_cache)
            level_data = hds.levels_cache[level]
            self.assertIn('k', level_data)
            self.assertIn('v', level_data)
            self.assertIn('num_slots', level_data)
    
    def test_hds_query(self):
        """Test HDS memory query functionality."""
        hds = HierarchicalDataStructure(
            memory_state=self.memory_state,
            num_levels=3,
            model_dim=self.config['model_dim']
        )
        
        # Construct hierarchy first
        hds.construct_hierarchy()
        
        # Create random query
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        query = torch.randn(batch_size, seq_len, model_dim,
                          dtype=self.config['dtype'], device=self.device)
        
        # Query memory at top level (level -1)
        k_level, v_level = hds.query_memory(query, level=-1)
        
        # Check that k and v are returned
        self.assertIsNotNone(k_level)
        self.assertIsNotNone(v_level)
        
        # Check dimensions (should have batch dimension)
        self.assertEqual(len(k_level.shape), 3)  # [batch, num_slots, k_dim]
        self.assertEqual(len(v_level.shape), 3)  # [batch, num_slots, v_dim]
        
        # Check batch dimension matches
        self.assertEqual(k_level.shape[0], batch_size)
        self.assertEqual(v_level.shape[0], batch_size)
        
        # Get level info
        level_info = hds.get_level_info(-1)
        self.assertIn('num_slots', level_info)
        self.assertIn('k_shape', level_info)
        self.assertIn('v_shape', level_info)


class TestMultiMemoryAttention(unittest.TestCase):
    """Tests for Multi-Memory Attention component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
        
        # Create memory state and HDS
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,
            'dtype': self.config['dtype']
        }
        
        self.memory_state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
        
        self.hds = HierarchicalDataStructure(
            memory_state=self.memory_state,
            num_levels=3,
            model_dim=self.config['model_dim']
        )
        self.hds.construct_hierarchy()
    
    def test_multi_memory_attention_forward(self):
        """Test MultiMemoryAttention forward pass."""
        attention = MultiMemoryAttention(
            model_dim=self.config['model_dim'],
            num_heads=self.config['num_heads']
        )
        
        # Create random query
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        query = torch.randn(batch_size, seq_len, model_dim,
                          dtype=self.config['dtype'], device=self.device)
        
        # Forward pass
        context = attention(query, self.hds, self.memory_state)
        
        # Check output dimensions
        self.assertEqual(context.shape, (batch_size, seq_len, model_dim))
        
        # Check that output is not NaN or Inf
        self.assertFalse(torch.isnan(context).any())
        self.assertFalse(torch.isinf(context).any())


class TestMMRecBlock(unittest.TestCase):
    """Tests for MM-Rec Block component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
        
        # Create memory state
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,
            'dtype': self.config['dtype']
        }
        
        self.memory_state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
    
    def test_mm_rec_block_forward(self):
        """Test MMRecBlock forward pass."""
        block = MMRecBlock(
            model_dim=self.config['model_dim'],
            inner_dim=self.config['inner_dim'],
            num_heads=self.config['num_heads'],
            num_memories=self.config['num_memories'],
            mem_dim=self.config['mem_dim'],
            ffn_dim=self.config['model_dim'] * 4,
            dropout=0.1
        )
        
        # Create random input
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        x = torch.randn(batch_size, seq_len, model_dim,
                       dtype=self.config['dtype'], device=self.device)
        
        # Forward pass
        x_out, updated_state = block(x, self.memory_state)
        
        # Check output dimensions
        self.assertEqual(x_out.shape, (batch_size, seq_len, model_dim))
        
        # Check that output is not NaN or Inf
        self.assertFalse(torch.isnan(x_out).any())
        self.assertFalse(torch.isinf(x_out).any())
        
        # Check that state is returned
        self.assertIsNotNone(updated_state)
        self.assertIsInstance(updated_state, MemoryState)
        
        # Check that input and output have same shape
        self.assertEqual(x.shape, x_out.shape)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TEST_CONFIG
        self.device = torch.device('cpu')
    
    def test_end_to_end_flow(self):
        """Test end-to-end flow through multiple components."""
        # Create memory state
        short_term_config = {
            'k_dim': self.config['model_dim'],
            'v_dim': self.config['model_dim'],
            'num_slots': self.config['seq_len'],
            'dtype': self.config['dtype']
        }
        
        long_term_config = {
            'k_dim': self.config['mem_dim'],
            'v_dim': self.config['mem_dim'],
            'num_slots': 1024,
            'dtype': self.config['dtype']
        }
        
        memory_state = MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=self.device
        )
        
        # Create HDS
        hds = HierarchicalDataStructure(
            memory_state=memory_state,
            num_levels=3,
            model_dim=self.config['model_dim']
        )
        hds.construct_hierarchy()
        
        # Create MDI
        mdi = MemoryDecayIntegration(
            model_dim=self.config['model_dim'],
            inner_dim=self.config['inner_dim']
        )
        
        # Create attention
        attention = MultiMemoryAttention(
            model_dim=self.config['model_dim'],
            num_heads=self.config['num_heads']
        )
        
        # Create block
        block = MMRecBlock(
            model_dim=self.config['model_dim'],
            inner_dim=self.config['inner_dim'],
            num_heads=self.config['num_heads'],
            num_memories=self.config['num_memories'],
            mem_dim=self.config['mem_dim']
        )
        
        # Create input
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        model_dim = self.config['model_dim']
        
        x = torch.randn(batch_size, seq_len, model_dim,
                       dtype=self.config['dtype'], device=self.device)
        
        # Test MDI
        z_t = torch.randn(batch_size, seq_len, model_dim,
                         dtype=self.config['dtype'], device=self.device)
        h_prev = x
        h_new, gamma = mdi(z_t, h_prev)
        self.assertEqual(h_new.shape, x.shape)
        
        # Test attention
        query = h_new
        context = attention(query, hds, memory_state)
        self.assertEqual(context.shape, x.shape)
        
        # Test block
        x_out, updated_state = block(x, memory_state)
        self.assertEqual(x_out.shape, x.shape)
        self.assertIsNotNone(updated_state)


if __name__ == '__main__':
    unittest.main()

