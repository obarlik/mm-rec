
"""
LSH Router (JAX)
Implements Locality Sensitive Hashing (SimHash) for ultra-fast routing.
Replaces expensive Top-K sorting with hashing.
"""

import jax
import jax.numpy as jnp
from jax import random

class LSHRouter:
    """
    Router that uses Locality Sensitive Hashing (Random Projections).
    Maps input to clusters using binary codes from random hyperplanes.
    Complexity: O(dim * log(num_clusters)) instead of O(dim * num_clusters).
    """
    
    def __init__(self, key, input_dim, num_clusters):
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        
        # We need log2(num_clusters) bits to address all clusters
        # e.g., 64 clusters -> 6 bits
        self.num_bits = int(jnp.ceil(jnp.log2(num_clusters)))
        
        # Random hyperplanes: [input_dim, num_bits]
        # This is much smaller than [input_dim, num_clusters]
        # e.g., 1024x6 vs 1024x64 -> 10x less computation
        self.hyperplanes = random.normal(key, (input_dim, self.num_bits))

    def route(self, x):
        """
        Hash inputs to clusters.
        
        Args:
            x: Input tensor [batch, dim]
            
        Returns:
            indices: Hash bucket indices [batch, 1] (Single expert per input for pure LSH)
            gates: Dummy gates (LSH is hard routing often, set to 1.0)
        """
        # 1. Random Projection
        # [batch, input_dim] @ [input_dim, num_bits] -> [batch, num_bits]
        projections = jnp.dot(x, self.hyperplanes)
        
        # 2. Sign hashing (SimHash)
        # Convert signs to bits: positive -> 1, negative -> 0
        bits = (projections > 0).astype(jnp.int32)
        
        # 3. Convert bits to integer index
        # e.g., [1, 0, 1] -> 1*4 + 0*2 + 1*1 = 5
        # Powers of 2: [1, 2, 4, 8, ...]
        powers = 2 ** jnp.arange(self.num_bits)
        
        # Dot product with powers to get integer index
        # [batch, num_bits] @ [num_bits] -> [batch]
        hashes = jnp.dot(bits, powers)
        
        # Expand dims for consistency with top-k interface [batch, 1]
        indices = hashes[:, None]
        
        # LSH is "Hard Routing", so gate weight is 1.0
        gates = jnp.ones_like(indices, dtype=jnp.float32)
        
        return indices, gates

def init_lsh_router(key, input_dim, num_clusters):
    """Functional initialization helper."""
    return LSHRouter(key, input_dim, num_clusters)
