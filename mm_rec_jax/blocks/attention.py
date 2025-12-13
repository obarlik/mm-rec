import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from ..core.hds import HDS
from ..core.memory_state import MemoryState

class MultiMemoryAttention(nn.Module):
    """
    Multi-Memory Attention Mechanism (JAX/Flax).
    
    Queries the hierarchical memory structure using the current hidden state.
    """
    model_dim: int
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    
    def setup(self):
        self.q_proj = nn.Dense(self.model_dim, use_bias=False)
        self.out_proj = nn.Dense(self.model_dim, use_bias=False)
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, 
                 h_t: jnp.ndarray, 
                 memory_state: MemoryState,
                 q_input: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        """
        Args:
            h_t: Hidden state [Batch, Dim] (or [Batch, Seq, Dim])
            memory_state: Current MemoryState
            q_input: Optional projected query from block [Batch, Seq, Dim]
            training: Dropout flag
            
        Returns:
            Attention output [Batch, Dim]
        """
        # 1. Project Query
        q_attention = self.q_proj(h_t) # [B, D]
        
        # CRITICAL: Mix with q_input if provided (Matches PyTorch)
        # q = q_attn + 0.5 * q_input
        if q_input is not None:
             query = q_attention + 0.5 * q_input
        else:
             query = q_attention
        
        # 2. Reshape for Multi-Head
        # [B, NumHeads, HeadDim]
        # In this simplified version, we'll assume flat attention first or standard MH
        # Let's stick to standard [B, H, D]
        
        # 3. Get Memory (Keys/Values) from HDS
        # For simplicity in this port, we default to Level 0 (Full Memory)
        # or implement logic to choose level. 
        # MM-Rec design: usually queries a specific level or all levels.
        # Let's assume Level 0 for high fidelity to current simple tests.
        
        hierarchy = HDS.construct_hierarchy(memory_state)
        k, v = HDS.query_memory(hierarchy, query, level=0)
        
        # k, v are [B, Slots, Dim] OR [Slots, Dim] (unbatched)
        
        # Handle unbatched case (e.g. during init)
        if k.ndim == 2:
            k = k[None, ...] # [1, S, D]
            v = v[None, ...]
            
        # query is [B, 1, Dim] (if h_t was [B, D])
        
        if query.ndim == 2:
            query = query[:, None, :] # [B, 1, D]
            
        # 4. Dot Product Attention
        # Q: [B, 1, D]
        # K: [B, S, D] -> K.T: [B, D, S]
        # Attn: [B, 1, S]
        
        scores = jnp.matmul(query, k.transpose((0, 2, 1))) * self.scale
        probs = nn.softmax(scores, axis=-1)
        
        output = jnp.matmul(probs, v) # [B, 1, D]
        
        # Remove seq dim if it was added
        if h_t.ndim == 2:
            output = output.squeeze(1)
            
        # 5. Output Projection
        output = self.out_proj(output)
        
        return output
