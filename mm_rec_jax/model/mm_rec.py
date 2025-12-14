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
    short_mem_len: int = 512
    long_mem_len: int = 512

    use_uboo: bool = False
    use_moe: bool = False
    lambda_p: float = 0.1 # Penalty scaling factor

    def setup(self):
        # Tied Weights: Embedding and LM Head share the same matrix
        self.shared_embed = SharedEmbedding(self.vocab_size, self.model_dim)
        
        self.blocks = [
            MMRecBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                ffn_dim=self.model_dim * 4,
                dropout_rate=self.dropout_rate,
                use_uboo=self.use_uboo,
                use_moe=self.use_moe, # Pass MoE flag
                name=f'block_{i}'
            ) for i in range(self.num_layers)
        ]
        self.norm_final = nn.RMSNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, 
                 x: jnp.ndarray, 
                 state: MemoryState, 
                 training: bool = False) -> Tuple[jnp.ndarray, MemoryState, jnp.ndarray]:
        """
        ...
        Returns: logits, state, aux_loss
        """
        # Embeddings (Mode='embed')
        h = self.shared_embed(x, mode='embed')
        h = self.dropout(h, deterministic=not training)
        
        total_aux_loss = jnp.array(0.0)
        
        # Pass through blocks
        for i, block in enumerate(self.blocks):
            # Block now returns (h, state, layer_aux_loss)
            h, state, layer_aux_loss = block(h, state, training=training)
            total_aux_loss += layer_aux_loss
            
        h = self.norm_final(h)
        
        # Output Head (Mode='projection') - Reuses embedding matrix
        logits = self.shared_embed(h, mode='projection')
        
        # Scale Total Aux Loss
        weighted_aux_loss = total_aux_loss * self.lambda_p
        
        return logits, state, weighted_aux_loss
    
    def initialize_state(self, batch_size: int):
        """Helper to create initial zero state (used by training loop)."""
        # Just create one instance, vmap will handle batching if we use pmap/vmap
        # Or create with batch dimension if manual.
        # For JAX scan, usually we pass simple unbatched state if vmapped, or batched if not.
        
        # Here we create a simple State PyTree
        # We invoke MemoryState.create (pure function usually)
        
        base_state = MemoryState.create(
            short_dim=self.model_dim,
            short_len=self.short_mem_len,
            long_dim=self.model_dim,
            long_len=self.long_mem_len
        )
        
        # Broadcast to batch dimensions using PyTree map
        if batch_size > 0:
            return jax.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), base_state)
            
        return base_state
