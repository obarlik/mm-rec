import jax
import jax.numpy as jnp
from flax import linen as nn

class SharedEmbedding(nn.Module):
    vocab_size: int
    model_dim: int
    
    @nn.compact
    def __call__(self, x, mode='embed'):
        # Define the embedding parameter explicitly to share it
        embedding = self.param('embedding', 
                               nn.initializers.normal(stddev=0.02), 
                               (self.vocab_size, self.model_dim))
        
        if mode == 'embed':
            # Gather: [B, T] -> [B, T, D]
            return jnp.take(embedding, x, axis=0)
        elif mode == 'projection':
            # Matmul: [B, T, D] @ [D, V] -> [B, T, V]
            # Output weight is embedding.T
            return jnp.dot(x, embedding.T)
