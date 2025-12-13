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
        Update Key and Value tensors for specified bank (full replacement).
        
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
    
    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get initial state (t=0) for a batch.
        
        Returns zero tensors representing initial memory state.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Tuple of (k_short_init, v_short_init, k_long_init, v_long_init)
            All tensors are zeros with appropriate shapes
        """
        # Short-term initial state: [batch, 1, model_dim] (single timestep)
        k_short_init = torch.zeros(
            batch_size, 1, self.short_term.k_dim,
            dtype=self.short_term.k.dtype,
            device=self.short_term.k.device
        )
        v_short_init = torch.zeros(
            batch_size, 1, self.short_term.v_dim,
            dtype=self.short_term.v.dtype,
            device=self.short_term.v.device
        )
        
        # Long-term initial state: [batch, num_memories, M, mem_dim]
        # Use existing long_term structure but create batch-sized zeros
        k_long_init = torch.zeros(
            batch_size, 1, self.long_term.num_slots, self.long_term.k_dim,
            dtype=self.long_term.k.dtype,
            device=self.long_term.k.device
        )
        v_long_init = torch.zeros(
            batch_size, 1, self.long_term.num_slots, self.long_term.v_dim,
            dtype=self.long_term.v.dtype,
            device=self.long_term.v.device
        )
        
        return k_short_init, v_short_init, k_long_init, v_long_init
    
    def update_state_sequential(
        self,
        bank_type: str,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        step: int
    ):
        """
        Update memory state at a specific sequence step.
        
        This is critical for short-term memory where we need to track
        state evolution across sequence steps.
        
        Args:
            bank_type: 'short' for short-term, 'long' for long-term
            new_k: New Key tensor for this step [batch, k_dim] or [batch, 1, k_dim]
            new_v: New Value tensor for this step [batch, v_dim] or [batch, 1, v_dim]
            step: Sequence step index (0-indexed)
        """
        if bank_type == 'short':
            # Ensure correct shape: [batch, k_dim] -> [batch, 1, k_dim]
            if new_k.dim() == 2:
                new_k = new_k.unsqueeze(1)
            if new_v.dim() == 2:
                new_v = new_v.unsqueeze(1)
            
            # Update at specific step: bank.k[step] = new_k
            # MemoryBank stores [num_slots, k_dim], but we need [batch, num_slots, k_dim]
            batch_size = new_k.shape[0]
            
            # Check current shape of bank tensors
            k_shape = self.short_term.k.data.shape
            v_shape = self.short_term.v.data.shape
            
            # If bank tensors are 2D [num_slots, k_dim], we need to handle per-batch
            # For now, store batch-specific state by creating a buffer
            # In practice, MemoryBank should support batch dimension
            
            if len(k_shape) == 2:
                # 2D tensor: [num_slots, k_dim] - no batch dimension
                # Update single slot (for first batch only, or average)
                if step < self.short_term.num_slots:
                    # Take mean across batch if multiple batches
                    if batch_size > 1:
                        new_k_mean = new_k.mean(dim=0, keepdim=True)  # [1, k_dim]
                        new_v_mean = new_v.mean(dim=0, keepdim=True)  # [1, v_dim]
                    else:
                        new_k_mean = new_k.squeeze(1)  # [k_dim]
                        new_v_mean = new_v.squeeze(1)  # [v_dim]
                    
                    self.short_term.k.data[step:step+1, :] = new_k_mean.to(self.short_term.k.device)
                    self.short_term.v.data[step:step+1, :] = new_v_mean.to(self.short_term.v.device)
                else:
                    raise ValueError(f"Step {step} exceeds num_slots {self.short_term.num_slots}")
            elif len(k_shape) == 3:
                # 3D tensor: [batch, num_slots, k_dim] - has batch dimension
                if k_shape[0] != batch_size:
                    # Resize to match batch size
                    old_k = self.short_term.k.data
                    old_v = self.short_term.v.data
                    self.short_term.k.data = torch.zeros(
                        batch_size, self.short_term.num_slots, self.short_term.k_dim,
                        dtype=old_k.dtype, device=old_k.device
                    )
                    self.short_term.v.data = torch.zeros(
                        batch_size, self.short_term.num_slots, self.short_term.v_dim,
                        dtype=old_v.dtype, device=old_v.device
                    )
                
                # Update at step position
                if step < self.short_term.num_slots:
                    self.short_term.k.data[:, step:step+1, :] = new_k.to(self.short_term.k.device)
                    self.short_term.v.data[:, step:step+1, :] = new_v.to(self.short_term.v.device)
                else:
                    raise ValueError(f"Step {step} exceeds num_slots {self.short_term.num_slots}")
            else:
                raise ValueError(f"Unexpected tensor shape: {k_shape}")
                
        elif bank_type == 'long':
            # Long-term memory update (less frequent, typically at block level)
            # For now, use full replacement
            if new_k.dim() == 3:  # [batch, num_slots, k_dim]
                self.long_term.k.data = new_k.to(self.long_term.k.device)
                self.long_term.v.data = new_v.to(self.long_term.v.device)
            else:
                # Single step update - update all slots with same value
                batch_size = new_k.shape[0]
                if new_k.dim() == 2:
                    new_k = new_k.unsqueeze(1).expand(-1, self.long_term.num_slots, -1)
                    new_v = new_v.unsqueeze(1).expand(-1, self.long_term.num_slots, -1)
                self.long_term.k.data = new_k.to(self.long_term.k.device)
                self.long_term.v.data = new_v.to(self.long_term.v.device)
        else:
            raise ValueError(f"Unknown bank_type: {bank_type}. Use 'short' or 'long'.")
    
    def get_state_at_step(self, bank_type: str, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get memory state at a specific sequence step.
        
        Args:
            bank_type: 'short' for short-term, 'long' for long-term
            step: Sequence step index (0-indexed)
        
        Returns:
            Tuple of (k_step, v_step) tensors for the specified step
        """
        if bank_type == 'short':
            k, v = self.get_state('short')
            # k: [num_slots, k_dim] or [batch, num_slots, k_dim]
            if k.dim() == 2:
                # No batch dimension
                k_step = k[step:step+1, :]  # [1, k_dim]
                v_step = v[step:step+1, :]  # [1, v_dim]
            else:
                # Has batch dimension
                k_step = k[:, step:step+1, :]  # [batch, 1, k_dim]
                v_step = v[:, step:step+1, :]  # [batch, 1, v_dim]
            return k_step, v_step
        elif bank_type == 'long':
            # Long-term memory doesn't have step-wise access
            # Return full memory
            return self.get_state('long')
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
    def update_state_chunk(
        self,
        bank_type: str,
        new_k_chunk: torch.Tensor,
        new_v_chunk: torch.Tensor,
        start_step: int
    ):
        """
        Update memory state efficiently for a chunk of steps (bulk update).
        
        Args:
            bank_type: 'short' or 'long'
            new_k_chunk: [batch, chunk_len, k_dim]
            new_v_chunk: [batch, chunk_len, v_dim]
            start_step: Starting sequence step index
        """
        if bank_type == 'short':
             # Ensure correct shape
            chunk_len = new_k_chunk.shape[1]
            end_step = start_step + chunk_len
            batch_size = new_k_chunk.shape[0]
            
            # Check shape of bank
            k_shape = self.short_term.k.data.shape
            
            if len(k_shape) == 2:
                # 2D case: [num_slots, k_dim] (No batching support implies single batch or shared)
                if end_step <= self.short_term.num_slots:
                    # Take mean if batched input for non-batched bank
                     if batch_size > 1:
                        new_k_mean = new_k_chunk.mean(dim=0)
                        new_v_mean = new_v_chunk.mean(dim=0)
                     else:
                        new_k_mean = new_k_chunk.squeeze(0)
                        new_v_mean = new_v_chunk.squeeze(0)
                        
                     self.short_term.k.data[start_step:end_step, :] = new_k_mean.to(self.short_term.k.device)
                     self.short_term.v.data[start_step:end_step, :] = new_v_mean.to(self.short_term.v.device)
            elif len(k_shape) == 3:
                # 3D case: [batch, num_slots, k_dim]
                if k_shape[0] != batch_size:
                    # Resize
                    old_k = self.short_term.k.data
                    old_v = self.short_term.v.data
                    self.short_term.k.data = torch.zeros(
                        batch_size, self.short_term.num_slots, self.short_term.k_dim,
                        dtype=old_k.dtype, device=old_k.device
                    )
                    self.short_term.v.data = torch.zeros(
                        batch_size, self.short_term.num_slots, self.short_term.v_dim,
                        dtype=old_v.dtype, device=old_v.device
                    )
                
                if end_step <= self.short_term.num_slots:
                    self.short_term.k.data[:, start_step:end_step, :] = new_k_chunk.to(self.short_term.k.device)
                    self.short_term.v.data[:, start_step:end_step, :] = new_v_chunk.to(self.short_term.v.device)
                else:
                    # Handle overflow (wrap-around not implemented for simple linear scan, assume truncated or standard seq)
                    # Just clip to max slots
                     valid_len = self.short_term.num_slots - start_step
                     if valid_len > 0:
                        self.short_term.k.data[:, start_step:start_step+valid_len, :] = new_k_chunk[:, :valid_len, :].to(self.short_term.k.device)
                        self.short_term.v.data[:, start_step:start_step+valid_len, :] = new_v_chunk[:, :valid_len, :].to(self.short_term.v.device)
            else:
                 raise ValueError(f"Unexpected tensor shape: {k_shape}")
        else:
            raise NotImplementedError("Bulk update for long-term memory not implemented yet.")


