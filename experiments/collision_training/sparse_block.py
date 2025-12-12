
"""
Sparse Block (JAX)
Simulates a layer with multiple independent clusters (experts).
Implements both Dense (Baseline) and Sparse (Collision-Based) computation.
"""

import jax
import jax.numpy as jnp
from jax import random

class SparseClusterBlock:
    """
    A block containing multiple neuron clusters.
    Can be executed densely (all clusters) or sparsely (collided clusters only).
    """
    
    def __init__(self, key, num_clusters, input_dim, hidden_dim):
        self.num_clusters = num_clusters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize parameters for ALL clusters
        # Stacked weights: [num_clusters, input_dim, hidden_dim]
        # This allows vmap over clusters
        k1, k2 = random.split(key)
        self.w_in = random.normal(k1, (num_clusters, input_dim, hidden_dim)) * 0.02
        self.w_out = random.normal(k2, (num_clusters, hidden_dim, input_dim)) * 0.02

    def forward_dense(self, x):
        """
        Baseline: Process input with ALL clusters (Brute Force).
        x: [batch, input_dim]
        """
        # Replicate x for all clusters: [batch, num_clusters, input_dim]
        # But wait, usually MOE means different inputs go to different experts via routing.
        # For simplicity in Dense benchmark: We compute ALL experts for ALL inputs and average them.
        # This represents the cost of a dense layer with same parameter count.
        
        # More realistic dense equivalent: a single large layer.
        # But to compare apples-to-apples (same architecture, just activation difference):
        
        # vmap over clusters
        def cluster_fwd(w_in, w_out, x_in):
            # MLP: x -> hidden -> x
            h = jnp.dot(x_in, w_in)
            h = jax.nn.relu(h)
            out = jnp.dot(h, w_out)
            return out

        # Apply all clusters to the same input x (broadcasting x)
        # x: [batch, input_dim] -> [batch, num_clusters, input_dim]
        batch_size = x.shape[0]
        
        # We want to compute: Output = Sum(Expert_i(x)) for i in 0..N
        
        # Map over batch
        def process_sample(x_s):
            # Map over clusters
            outs = jax.vmap(cluster_fwd, (0, 0, None))(self.w_in, self.w_out, x_s)
            # outs: [num_clusters, input_dim]
            return jnp.sum(outs, axis=0)
            
        return jax.vmap(process_sample)(x)

    def forward_sparse(self, x, router_indices, router_gates):
        """
        Collision-Based: Process input ONLY with collided clusters.
        x: [batch, input_dim]
        router_indices: [batch, top_k]
        router_gates: [batch, top_k]
        """
        batch_size, top_k = router_indices.shape
        
        # We need to gather the weights for the selected clusters.
        # w_in: [num_clusters, input_dim, hidden_dim]
        
        # Gather weights for each sample's top-k clusters
        # goal: [batch, top_k, input_dim, hidden_dim]
        
        # JAX optimized gather
        # We used take or indexing
        selected_w_in = self.w_in[router_indices]   # [batch, top_k, in, hid]
        selected_w_out = self.w_out[router_indices] # [batch, top_k, hid, in]
        
        # Now we process each sample's top-k interactions
        # This is strictly [batch, top_k] computation instead of [batch, num_clusters]
        
        def process_sample(x_s, w_in_s, w_out_s, gates_s):
            # x_s: [input_dim]
            # w_in_s: [top_k, input_dim, hidden_dim]
            # w_out_s: [top_k, hidden_dim, input_dim]
            
            # Optimized Einsum Implementation (No internal vmap)
            
            # 1. Input Projection: x * W_in
            # [input_dim] dot [top_k, input_dim, hidden_dim] -> [top_k, hidden_dim]
            h = jnp.einsum('i,kih->kh', x_s, w_in_s)
            
            # Activation
            h = jax.nn.relu(h)
            
            # 2. Output Projection: h * W_out
            # [top_k, hidden_dim] dot [top_k, hidden_dim, input_dim] -> [top_k, input_dim]
            outs = jnp.einsum('kh,khi->ki', h, w_out_s)
            
            # Weight by gates (Collision intensity)
            gate_broadcast = gates_s[:, None] # [top_k, 1]
            weighted_outs = outs * gate_broadcast
            
            return jnp.sum(weighted_outs, axis=0) # Sum over top experts
            
        return jax.vmap(process_sample)(x, selected_w_in, selected_w_out, router_gates)

def init_block(key, num_clusters, input_dim, hidden_dim):
    return SparseClusterBlock(key, num_clusters, input_dim, hidden_dim)
