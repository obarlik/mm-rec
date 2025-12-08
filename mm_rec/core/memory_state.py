"""
MM-Rec Memory State Management
Manages short-term (h_t) and long-term (M) memory banks
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MemoryBankConfig:
    """Configuration for a memory bank."""
    k_dim: int
    v_dim: int
    num_slots: int
    dtype: torch.dtype = torch.bfloat16


class MemoryBank(nn.Module):
    """
    Single memory bank unit (short-term or long-term).
    
    Represents a memory bank with Key-Value pairs.
    
    Args:
        k_dim: Dimension of keys
        v_dim: Dimension of values
        num_slots: Number of memory slots
        dtype: Data type for tensors (default: bfloat16)
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        k_dim: int,
        v_dim: int,
        num_slots: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_slots = num_slots
        self.dtype = dtype
        self.device = device
        
        # Initialize Key and Value tensors
        self.k = nn.Parameter(
            torch.zeros(num_slots, k_dim, dtype=dtype, device=device),
            requires_grad=True
        )
        self.v = nn.Parameter(
            torch.zeros(num_slots, v_dim, dtype=dtype, device=device),
            requires_grad=True
        )
        
        # Initialize with small random values
        self.initialize_bank(num_slots)
    
    def initialize_bank(self, num_slots: Optional[int] = None):
        """
        Initialize bank with Gaussian distribution.
        
        Args:
            num_slots: Number of slots to initialize (if None, uses self.num_slots)
        """
        if num_slots is None:
            num_slots = self.num_slots
        
        # Initialize with small Gaussian noise
        with torch.no_grad():
            self.k.data.normal_(mean=0.0, std=0.02)
            self.v.data.normal_(mean=0.0, std=0.02)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Key and Value tensors.
        
        Returns:
            Tuple of (k, v) tensors
        """
        return self.k, self.v
    
    def to_device(self, device: torch.device):
        """Move all tensors to specified device."""
        self.device = device
        self.k.data = self.k.data.to(device)
        self.v.data = self.v.data.to(device)


class MemoryState(nn.Module):
    """
    Manages overall memory state of the model.
    
    Contains both short-term (h_t) and long-term (M) memory banks.
    
    Short-term memory: [batch, seq_len, hidden_dim] - Per-token hidden states
    Long-term memory: [batch, num_memories, M, mem_dim] - Persistent memory matrix
    where M << seq_len (typically M=1024 for 32K+ sequences)
    
    Args:
        short_term_config: Configuration dict for short-term memory bank
        long_term_config: Configuration dict for long-term memory bank
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        short_term_config: Dict,
        long_term_config: Dict,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device
        
        # Create short-term memory bank
        short_config = MemoryBankConfig(**short_term_config)
        self.short_term = MemoryBank(
            k_dim=short_config.k_dim,
            v_dim=short_config.v_dim,
            num_slots=short_config.num_slots,
            dtype=short_config.dtype,
            device=device
        )
        
        # Create long-term memory bank
        long_config = MemoryBankConfig(**long_term_config)
        self.long_term = MemoryBank(
            k_dim=long_config.k_dim,
            v_dim=long_config.v_dim,
            num_slots=long_config.num_slots,
            dtype=long_config.dtype,
            device=device
        )
    
    def get_state(self, bank_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Key and Value tensors for specified bank type.
        
        Args:
            bank_type: 'short' for short-term, 'long' for long-term
        
        Returns:
            Tuple of (k, v) tensors
        """
        if bank_type == 'short':
            return self.short_term()
        elif bank_type == 'long':
            return self.long_term()
        else:
            raise ValueError(f"Unknown bank_type: {bank_type}. Use 'short' or 'long'.")
    
    def update_state(
        self,
        bank_type: str,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ):
        """
        Update Key and Value tensors for specified bank.
        
        Args:
            bank_type: 'short' for short-term, 'long' for long-term
            new_k: New Key tensor
            new_v: New Value tensor
        """
        if bank_type == 'short':
            self.short_term.k.data = new_k.to(self.short_term.k.device)
            self.short_term.v.data = new_v.to(self.short_term.v.device)
        elif bank_type == 'long':
            self.long_term.k.data = new_k.to(self.long_term.k.device)
            self.long_term.v.data = new_v.to(self.long_term.v.device)
        else:
            raise ValueError(f"Unknown bank_type: {bank_type}. Use 'short' or 'long'.")
    
    def to_device(self, device: torch.device):
        """Move all memory banks to specified device."""
        self.device = device
        self.short_term.to_device(device)
        self.long_term.to_device(device)
    
    def forward(self, bank_type: str = 'short') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - get state for specified bank.
        
        Args:
            bank_type: 'short' or 'long'
        
        Returns:
            Tuple of (k, v) tensors
        """
        return self.get_state(bank_type)

