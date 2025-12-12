
"""
Collision Router (JAX)
Implements 'Collision Detection' logic to route inputs to specific neuron clusters.
"""

import jax
import jax.numpy as jnp
from jax import random

class CollisionRouter:
    """
    Router that detects which clusters 'collide' (resonate) with the input.
    Uses Top-K gating mechanism.
    """
    
    def __init__(self, key, input_dim, num_clusters, top_k=4):
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.top_k = top_k
        
        # Router weights: [input_dim, num_clusters]
        # These project input to 'cluster space' to check for collision
        self.w_gate = random.normal(key, (input_dim, num_clusters)) * 0.02

    def route(self, x):
        """
        Detect collisions for input x.
        
        Args:
            x: Input tensor [batch, dim]
            
        Returns:
            indices: Indices of collided clusters [batch, top_k]
            gates: Activation strength (collision intensity) [batch, top_k]
        """
        # 1. Project to cluster space
        # logits: [batch, num_clusters]
        logits = jnp.dot(x, self.w_gate)
        
        # 2. Add noise for training stability (optional, skipped for simplicity)
        
        # 3. Detect Top-K Collisions
        # gates: Top-K values, indices: Top-K indices
        gates, indices = jax.lax.top_k(logits, self.top_k)
        
        # 4. Softmax over Top-K (Collision Intensity Normalization)
        # We only care about relative strength among the collided ones
        gates = jax.nn.softmax(gates, axis=-1)
        
        return indices, gates

def init_router(key, input_dim, num_clusters, top_k=4):
    """Functional initialization helper."""
    return CollisionRouter(key, input_dim, num_clusters, top_k)
