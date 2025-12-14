import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any, Optional

class JaxLSHRouter(nn.Module):
    """
    JAX-Native LSH Router with Fixed Capacity (TPU/GPU Friendly).
    
    Implements:
    1. SimHash (Random Projection) for hashing
    2. Top-K Selection
    3. Fixed Capacity Enforcement (Dropping overflow)
    """
    model_dim: int
    num_experts: int = 8
    capacity_factor: float = 1.0 # 1.0 = Perfectly balanced expectation
    top_k: int = 2
    
    def setup(self):
        # LSH Random Projection (Fixed, distinct for each expert bit)
        # We need log2(num_experts) bits to hash to an expert
        # E.g. 8 experts -> 3 bits. 
        # But SimHash usually maps to a larger space. 
        # Here we use a learnable router or simple projection for simplicity.
        # "SimHash" usually implies fixed random weights.
        self.hash_proj = nn.Dense(self.num_experts, use_bias=False, kernel_init=nn.initializers.orthogonal())

    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Input [Batch, Seq, Dim]
            
        Returns:
            combine_weights: [Batch, Seq, TopK]
            dispatch_mask: [Batch, Seq, TopK, Num_Experts, Capacity] (Huge simplified tensor for einsum)
                           OR we return the indices for manual gather/scatter.
            
            Actually for JAX efficiency, we return the 'dispatch_tensor' directly.
        """
        batch_size, seq_len, dim = x.shape
        num_tokens = batch_size * seq_len
        
        # 1. Routing Logits
        logits = self.hash_proj(x)
        if training:
            logits += jax.random.normal(self.make_rng('dropout'), logits.shape) * 0.1
            
        # 2. Top-K Selection
        # logits: [B, S, E]
        gate_logits, expert_indices = jax.lax.top_k(logits, self.top_k)
        gate_weights = nn.softmax(gate_logits, axis=-1) # [B, S, K]
        
        # 3. Capacity Calculation
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
        
        # 4. Create Index within Expert Buffer
        # We need to assign each selected token to a slot [0..Capacity-1] in the chosen expert.
        
        # Mask: [B, S, K, E] (One-hot for each of the Top-K choices)
        # expand indices to use one_hot
        expert_mask = jax.nn.one_hot(expert_indices, self.num_experts) # [B, S, K, E]
        
        # Flatten batch/seq for counting -> [N, K, E]
        expert_mask_flat = expert_mask.reshape(-1, self.top_k, self.num_experts)
        
        # Priority/Index: CumSum along token dimension (N)
        # "Which number token am I for this expert?"
        # [N, K, E]
        token_priority = jnp.cumsum(expert_mask_flat, axis=0)
        
        # Valid Mask: Is priority <= Capacity?
        # [N, K, E]
        valid_mask_flat = (token_priority > 0) & (token_priority <= capacity)
        
        # Dispatch Indices (Slot in Capacity) - 0-indexed (priority - 1)
        # [N, K, E]
        dispatch_idx_flat = (token_priority - 1) * valid_mask_flat
        
        # Eliminate 'E' dimension to get direct index for the chosen expert
        # Since each K picks exactly one E, we can sum along E (others are 0)
        # But we need to know WHICH expert.
        # We process experts in parallel, so we need a tensor [N, K, E, C] or similar.
        # Efficient approach: Einsum Dispatch
        
        # Combine Mask [N, K, E] and Location One-Hot [N, K, E, C]
        # This is memory explosive.
        # Better: ScatterND or Matmul Dispatch.
        
        # MATMUL DISPATCH STRATEGY (Google/Switch Transformer style)
        # Dispatch Tensor: [N, E*C] or [N, E, C]
        # Let's target [Num_Experts, Capacity, Dim] as input to experts.
        
        # We create a mapping: M [N, E, C]
        # M[n, e, c] = 1 if token n goes to expert e at slot c.
        # valid_mask: [N, K, E]
        # slot_idx: [N, K, E] (values 0..C-1)
        
        # Create One-Hot of Slot: [N, K, E, C]
        slot_one_hot = jax.nn.one_hot(dispatch_idx_flat.astype(jnp.int32), capacity)
        
        # Combine with Expert selection and Validity
        # D [N, K, E, C] = ExpertMask [N, K, E] * Valid [N, K, E] * Slot [N, K, E, C]
        # Reshape ExpertMask/Valid for broadcasting
        mask_expanded = (expert_mask_flat * valid_mask_flat)[..., None] # [N, K, E, 1]
        dispatch_tensor = mask_expanded * slot_one_hot # [N, K, E, C]
        
        # Sum over K (Top-K heads) -> [N, E, C]
        # A token is dispatched to multiple experts/slots defined by K
        dispatcher = jnp.sum(dispatch_tensor, axis=1) # [N, E, C]
        
        return dispatcher, gate_weights, expert_indices

class SparseMMRecBlock(nn.Module):
    """
    Mixture-of-Experts Block wrapping standard FFNs.
    """
    model_dim: int
    num_experts: int = 8
    capacity_factor: float = 1.0
    
    def setup(self):
        self.router = JaxLSHRouter(
            self.model_dim, 
            self.num_experts, 
            capacity_factor=self.capacity_factor
        )
        # Experts: We use Vmap over a single Dense layer definition to share logic but separate weights
        # Actually in Flax, we can define a layer and vmap its 'apply'.
        # Or store weights as [Experts, In, Out] and use dot_general.
        
        # Approach: Single Dense with Extra Dimension
        # Expert Weights: [Experts, Dim, Dim]
        self.expert_w1 = self.param('expert_w1', nn.initializers.xavier_uniform(), 
                                    (self.num_experts, self.model_dim, self.model_dim * 4))
        self.expert_w2 = self.param('expert_w2', nn.initializers.zeros, 
                                    (self.num_experts, self.model_dim * 4, self.model_dim))

    def __call__(self, x, training: bool = False):
        """
        x: [Batch, Seq, Dim]
        """
        B, S, D = x.shape
        x_flat = x.reshape(-1, D) # [N, D]
        
        # 1. Route
        # dispatcher: [N, E, C] (One-hot mapping token -> expert slot)
        # weights: [B, S, K]
        # indices: [B, S, K]
        dispatcher, gate_weights_bs, indices = self.router(x, training=training)
        
        # Flatten weights for combine: [N, K]
        gate_weights = gate_weights_bs.reshape(-1, gate_weights_bs.shape[-1])
        
        # 2. Dispatch (Scatter)
        # Input: [N, D]
        # Dispatcher: [N, E, C]
        # Output: [E, C, D]
        # Equation: e c d <- n d, n e c
        expert_inputs = jnp.einsum('nd, nec -> ecd', x_flat, dispatcher)
        
        # 3. Expert Computation (Parallel)
        # Input: [E, C, D]
        # W1: [E, D, H]
        # W2: [E, H, D]
        
        # Layer 1: [E, C, H] = [E, C, D] @ [E, D, H]
        h = jnp.einsum('ecd, edh -> ech', expert_inputs, self.expert_w1)
        h = nn.gelu(h)
        
        # Layer 2: [E, C, D] = [E, C, H] @ [E, H, D]
        expert_outputs = jnp.einsum('ech, ehd -> ecd', h, self.expert_w2)
        
        # 4. Combine (Gather)
        # Input: [E, C, D] (Expert results)
        # Dispatcher: [N, E, C] (Mapping back)
        # Output: [N, D] (Weighted sum of expert results)
        
        # Gather: [N, E, D] = [N, E, C] * [E, C, D] (Sum over C)
        # We retrieve the result for token N from expert E (at its assigned slot C)
        # Equation: n e d <- n e c, e c d
        unweighted_outputs = jnp.einsum('nec, ecd -> ned', dispatcher, expert_outputs)
        
        # Weighted Sum: Sum over Top-K experts
        # We need to match unweighted_outputs [N, Expert, D] with weights [N, K]
        # But 'unweighted_outputs' has values for ALL experts (sparse).
        
        # We need to select the specific experts chosen by Top-K.
        # indices: [N, K] (Indices of experts 0..E-1)
        # weights: [N, K]
        
        # Gather results for the K chosen experts
        # One-hot selection: [N, K, E]
        expert_selection_mask = jax.nn.one_hot(indices.reshape(-1, indices.shape[-1]), self.num_experts)
        
        # [N, K, D] = [N, E, D] * [N, K, E] (Select E)
        # Equation: n k d <- n e d, n k e
        top_k_outputs = jnp.einsum('ned, nke -> nkd', unweighted_outputs, expert_selection_mask)
        
        # Apply Gates
        # [N, K, D] * [N, K, 1]
        weighted_output = top_k_outputs * gate_weights[..., None]
        
        # Sum over K
        output_flat = jnp.sum(weighted_output, axis=1) # [N, D]
        
        # Reshape to [Batch, Seq, Dim]
        output = output_flat.reshape(B, S, D)
        
        return output
