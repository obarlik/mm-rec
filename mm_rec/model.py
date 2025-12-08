"""
MM-Rec Complete Model
24-layer architecture with embedding and output head
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .core.memory_state import MemoryState, MemoryBankConfig
from .blocks.mm_rec_block import MMRecBlock
from .blocks.mm_rec_block import RMSNorm


class MMRecModel(nn.Module):
    """
    Complete MM-Rec model architecture.
    
    Configuration (7B Model - REQUIRED):
    - hidden_dim: 4096 (D_hidden, REQUIRED)
    - num_layers: 24 (L_layer, REQUIRED)
    - max_seq_len: >= 32768 (N_sequence ≥ 32K, REQUIRED)
    - memory_size_M: 1024 (long-term memory size, M << seq_len)
    
    Args:
        vocab_size: Vocabulary size
        model_dim: Model dimension (hidden_dim, default: 4096)
        num_layers: Number of MM-Rec blocks (default: 24)
        num_heads: Number of attention heads
        num_memories: Number of memory banks
        mem_dim: Memory dimension
        max_seq_len: Maximum sequence length (default: 32768)
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 4096,
        num_layers: int = 24,
        num_heads: int = 8,
        num_memories: int = 1,
        mem_dim: Optional[int] = None,
        max_seq_len: int = 32768,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memories = num_memories
        self.mem_dim = mem_dim if mem_dim is not None else model_dim
        self.max_seq_len = max_seq_len
        self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, model_dim)
        
        # Initialize memory state
        # Short-term memory: [batch, seq_len, hidden_dim]
        # Long-term memory: [batch, num_memories, M, mem_dim] where M << seq_len
        M = 1024  # Long-term memory size (M << max_seq_len)
        
        # Use float32 for training stability (can be changed to bfloat16 for inference)
        memory_dtype = torch.float32
        
        short_term_config = {
            'k_dim': model_dim,
            'v_dim': model_dim,
            'num_slots': max_seq_len,  # Can hold full sequence
            'dtype': memory_dtype
        }
        
        long_term_config = {
            'k_dim': self.mem_dim,
            'v_dim': self.mem_dim,
            'num_slots': M,  # Fixed size M << seq_len
            'dtype': memory_dtype
        }
        
        # Create initial memory state (will be created per batch)
        self.memory_config = {
            'short_term': short_term_config,
            'long_term': long_term_config
        }
        
        # MM-Rec blocks (24 layers as per spec)
        self.blocks = nn.ModuleList([
            MMRecBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_memories=num_memories,
                mem_dim=self.mem_dim,
                ffn_dim=self.ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(model_dim)
        
        # Output head (language modeling)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Tie weights: output embedding = input embedding (common in LLMs)
        self.embedding.weight = self.lm_head.weight
        
        # Performance optimization flags
        self.use_gradient_checkpointing = False  # Can be enabled for memory efficiency
    
    def create_memory_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """
        Create a new MemoryState for a batch.
        
        Args:
            batch_size: Batch size
            device: Device to create state on
        
        Returns:
            MemoryState instance
        """
        return MemoryState(
            short_term_config=self.memory_config['short_term'],
            long_term_config=self.memory_config['long_term'],
            device=device
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through complete MM-Rec model.
        
        CRITICAL: Chunking support for long sequences (100K+).
        For sequences longer than chunk_size, the input is split into chunks
        and processed sequentially with memory state carry-over.
        
        This reduces memory from O(N) to O(B) where B is chunk_size.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_states: Optional list of MemoryState instances (one per layer)
            chunk_size: Chunk size for long sequences (default: 8192, or None to disable)
                       If seq_len > chunk_size, chunking is automatically enabled.
        
        Returns:
            logits: Language modeling logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Determine chunk size (default: 8192 for sequences > 32K)
        if chunk_size is None:
            # Auto-enable chunking for very long sequences
            if seq_len > 32768:
                chunk_size = 8192  # 8K chunks for 100K+ sequences
            else:
                chunk_size = None  # No chunking for shorter sequences
        
        # Create memory states if not provided
        if memory_states is None:
            memory_states = [
                self.create_memory_state(batch_size, device)
                for _ in range(self.num_layers)
            ]
        
        # CHUNKING: Process long sequences in chunks to reduce memory from O(N) to O(B)
        if chunk_size is not None and seq_len > chunk_size:
            # Split input into chunks
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            all_logits = []
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]  # [batch, chunk_len]
                
                # Embedding: Convert token IDs to embeddings
                x_chunk = self.embedding(chunk_input_ids)  # [batch, chunk_len, model_dim]
                
                # Forward through all MM-Rec blocks with carry-over memory states
                new_memory_states = []
                for i, block in enumerate(self.blocks):
                    # Enable checkpointing for deeper layers (saves more memory)
                    use_checkpointing = getattr(self, 'use_gradient_checkpointing', False)
                    if use_checkpointing and i >= len(self.blocks) // 2:  # Checkpoint deeper layers
                        from torch.utils.checkpoint import checkpoint
                        def block_forward(x_in, state_in):
                            return block(x_in, state_in, use_checkpointing=False)
                        x_chunk, updated_state = checkpoint(
                            block_forward, x_chunk, memory_states[i], use_reentrant=False
                        )
                    else:
                        x_chunk, updated_state = block(x_chunk, memory_states[i])
                    
                    # CRITICAL: Carry-over memory state to next chunk
                    memory_states[i] = updated_state
                    new_memory_states.append(updated_state)
                
                # Final normalization
                x_chunk = self.norm(x_chunk)  # [batch, chunk_len, model_dim]
                
                # Output head: Compute logits for this chunk
                logits_chunk = self.lm_head(x_chunk)  # [batch, chunk_len, vocab_size]
                all_logits.append(logits_chunk)
            
            # Concatenate all chunk logits
            logits = torch.cat(all_logits, dim=1)  # [batch, seq_len, vocab_size]
            
        else:
            # NO CHUNKING: Process entire sequence at once (for shorter sequences)
            # Embedding: Convert token IDs to embeddings
            x = self.embedding(input_ids)  # [batch, seq_len, model_dim]
            
            # Forward through all MM-Rec blocks
            # OPTIMIZATION: Use gradient checkpointing for memory efficiency
            # Can be enabled via model configuration
            new_memory_states = []
            for i, block in enumerate(self.blocks):
                # Enable checkpointing for deeper layers (saves more memory)
                use_checkpointing = getattr(self, 'use_gradient_checkpointing', False)
                if use_checkpointing and i >= len(self.blocks) // 2:  # Checkpoint deeper layers
                    from torch.utils.checkpoint import checkpoint
                    def block_forward(x_in, state_in):
                        return block(x_in, state_in, use_checkpointing=False)
                    x, updated_state = checkpoint(block_forward, x, memory_states[i], use_reentrant=False)
                else:
                    x, updated_state = block(x, memory_states[i])
                new_memory_states.append(updated_state)
            
            # Final normalization
            x = self.norm(x)  # [batch, seq_len, model_dim]
            
            # Output head: Compute logits
            logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

