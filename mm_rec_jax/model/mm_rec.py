import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List

from ..core.memory_state import MemoryState
from ..blocks.block import MMRecBlock
from .shared_embedding import SharedEmbedding

class MMRecModel(nn.Module):
    """
    MM-Rec Full Model (JAX/Flax).
    """
    vocab_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int = 32768
    dropout_rate: float = 0.1
    
    # Memory Config
    short_mem_len: int = 512 # Reduced from 2048 for benchmark stability
    long_mem_len: int = 512
    
    def setup(self):
        # Tied Weights: Embedding and LM Head share the same matrix
        self.shared_embed = SharedEmbedding(self.vocab_size, self.model_dim)
        
        self.blocks = [
            MMRecBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                ffn_dim=self.model_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}'
            ) for i in range(self.num_layers)
        ]
        self.norm_final = nn.RMSNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, 
                 x: jnp.ndarray, 
                 state: MemoryState, 
                 training: bool = False) -> Tuple[jnp.ndarray, MemoryState]:
        """
        ...
        """
        # Embeddings (Mode='embed')
        h = self.shared_embed(x, mode='embed')
        h = self.dropout(h, deterministic=not training)
        
        # Pass through blocks
        for i, block in enumerate(self.blocks):
            h, state = block(h, state, training=training)
            
        h = self.norm_final(h)
        
        # Output Head (Mode='projection') - Reuses embedding matrix
        logits = self.shared_embed(h, mode='projection')
        
        return logits, state
    
    def initialize_state(self, batch_size: int):
        """Helper to create initial zero state (used by training loop)."""
        # Just create one instance, vmap will handle batching if we use pmap/vmap
        # Or create with batch dimension if manual.
        # For JAX scan, usually we pass simple unbatched state if vmapped, or batched if not.
        
        # Here we create a simple State PyTree
        # We invoke MemoryState.create (pure function usually)
        
        return MemoryState.create(
            short_dim=self.model_dim,
            short_len=self.short_mem_len,
            long_dim=self.model_dim,
            long_len=self.long_mem_len
        )
