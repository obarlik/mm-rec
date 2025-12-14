import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any

from ..core.memory_state import MemoryState
from ..core.hds import HDS
from .attention import MultiMemoryAttention
from .sparse_block import SparseMMRecBlock

class MMRecBlock(nn.Module):
    """
    MM-Rec Block (JAX/Flax).
    Implements efficient gated recurrence using jax.lax.scan.
    """
    model_dim: int
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout_rate: float = 0.1
    use_uboo: bool = False # Unbiased Optimization flag
    use_moe: bool = False  # Mixture of Experts flag
    
    def setup(self):
        # ... (Previous code) ...
        
        # FFN Configuration
        if self.use_moe:
            # Replaces standard FFN with MoE Block
            self.moe_ffn = SparseMMRecBlock(
                model_dim=self.model_dim,
                num_experts=8, # Default, can be parameterized
                capacity_factor=1.25 # Slight over-capacity to reduce drops
            )
            # We still need dropout for the output of MoE
            self.ffn_drop2 = nn.Dropout(self.dropout_rate)
        else:
            # Standard Dense FFN
            self.ffn_dense1 = nn.Dense(self.ffn_dim, kernel_init=nn.initializers.xavier_uniform())
            self.ffn_act = nn.gelu
            self.ffn_drop1 = nn.Dropout(self.dropout_rate)
            self.ffn_dense2 = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
            self.ffn_drop2 = nn.Dropout(self.dropout_rate)
            
        self.dropout = nn.Dropout(self.dropout_rate)
        # Projections
        # Using Xavier Uniform to match PyTorch
        # Projections (HEM: Hardware Efficient Memory - Fused Kernel)
        # We fuse Q, K, V, Z into a single projection for efficiency
        # Output dim = 4 * model_dim
        self.W_fused = nn.Dense(self.model_dim * 4, kernel_init=nn.initializers.xavier_uniform())
        
        # Gating (Split for efficient MDI)
        # Matches PyTorch's Linear(2*dim, dim) by summing two projections
        # Using Xavier Uniform to match PyTorch explicit initialization
        self.W_g_z = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        self.W_g_h = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        
        self.W_g_mdi = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform()) # For gamma
        
        # Gamma (MDI)
        # PyTorch MDI W_gamma is: Linear -> GELU -> Linear -> Sigmoid
        self.W_gamma_1 = nn.Dense(self.model_dim // 4, kernel_init=nn.initializers.xavier_uniform())
        # Init Gamma 2 with Zeros and Negative Bias to start with small Gamma (~0.05)
        # This prevents exponential explosion in the recurrence (1 + gamma)^T
        self.W_gamma_2 = nn.Dense(
            self.model_dim, 
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -3.0, d)
        )
        
        # Context Modulation (MDI)
        # Modulates gamma based on context (Keys)
        self.W_gamma_context_1 = nn.Dense(self.model_dim // 4, kernel_init=nn.initializers.xavier_uniform())
        # Init context modulation to identity/zero initially
        self.W_gamma_context_2 = nn.Dense(
             self.model_dim,
             kernel_init=nn.initializers.zeros,
             bias_init=nn.initializers.zeros
        )
        
        # Attention
        self.attn = MultiMemoryAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            head_dim=self.model_dim // self.num_heads
        )
        
        # Norms
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        # Recurrence Norm (Critical for MDI stability)
        self.norm_recurrence = nn.RMSNorm()
        
        # UBOO Projections (Planning Error)
        if self.use_uboo:
            # Error Pred: h_new (detached) -> Error Space
            self.W_planning_error = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
            # Target: h_prev -> Error Space
            self.W_planning_target = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, 
                 x: jnp.ndarray, 
                 state: MemoryState, 
                 training: bool = False) -> Tuple[jnp.ndarray, MemoryState, jnp.ndarray]:
        """
        Forward pass with Scan.
        
        Args:
            x: Input [Batch, Seq, Dim]
            state: Functional MemoryState
            
        Returns:
            (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections (HEM: Fused Kernel)
        x_norm = self.norm1(x)
        
        # Single fused matmul [Batch, Seq, 4*Dim]
        fused_out = self.W_fused(x_norm)
        
        # Split into components
        # jnp.split is efficient in JAX (view-based or XLA fused)
        q, k, v, z = jnp.split(fused_out, 4, axis=-1)
        
        # Pre-compute Gamma (MDI)
        # Gamma = Sigmoid(W2(GELU(W1(z))))
        gamma_hidden = nn.gelu(self.W_gamma_1(z))
        gamma_base = nn.sigmoid(self.W_gamma_2(gamma_hidden))
        
        # Context Modulation (using k as context, matching PyTorch line 504)
        # modulation = Sigmoid(W_ctx(k))
        ctx_hidden = nn.gelu(self.W_gamma_context_1(k)) # k is context
        gamma_mod = nn.sigmoid(self.W_gamma_context_2(ctx_hidden))
        
        gamma = gamma_base * gamma_mod
        
        # Stability Clamp (Matches PyTorch MDI)
        gamma = jnp.clip(gamma, 1e-6, 1.0 - 1e-6)
        
        # Initial hidden state (zeros) [Batch, Dim]
        h0 = jnp.zeros((batch_size, self.model_dim))
        
        # Pre-compute Vectorized Gating Part 1 (z dependent)
        # This moves W_g_z out of scan for speed and fixes init issue
        gate_z_seq = self.W_g_z(z) # [Batch, Seq, Dim]
        
        # Force Init of W_g_h (h dependent)
        # Scan cannot initialize variables, so we run it once here
        _ = self.W_g_h(h0)
        # Force Init of Recurrence Norm
        _ = self.norm_recurrence(h0)
        
        # 2. Recurrence Scan Function
        # Scan carries: (h_prev)
        # Inputs: (gate_z_t, z_t, gamma_t)
        
        def scan_fn(carry, inputs):
            h_prev = carry
            gate_z_t, z_t, gamma_t = inputs
            
            # Gating Logic
            # gate_logits = W_g_z(z) + W_g_h(h)
            # W_g_z(z) is precomputed
            gate_logits = gate_z_t + self.W_g_h(h_prev)
            gate = nn.sigmoid(gate_logits)
            
            # MDI Formula:
            # h_tilde = (1 - gate) * h_prev + gate * z_t
            h_tilde = (1.0 - gate) * h_prev + gate * z_t
            
            # Final: h_new = h_tilde + gamma * h_prev
            h_t = h_tilde + gamma_t * h_prev
            
            # Stabilization: LayerNorm inside recurrence
            # This keeps h_t distribution stable, preventing drift to infinity
            h_t = self.norm_recurrence(h_t)
            
            # CRITICAL SAFETY: Clip hidden state to prevent exponential explosion
            # Since terms can sum > 1.0, this prevents infinity over long sequences
            h_t = jnp.clip(h_t, -100.0, 100.0)
            
            return h_t, h_t
        # scan arguments: (f, init, xs)
        # xs structure must match inputs of scan_fn
        xs = (gate_z_seq, z, gamma)
        
        # Transpose inputs to [Seq, Batch, Dim]
        gate_z_T = jnp.swapaxes(gate_z_seq, 0, 1)
        z_T = jnp.swapaxes(z, 0, 1)
        gamma_T = jnp.swapaxes(gamma, 0, 1)
        
        # Run scan
        _, h_sequence_T = jax.lax.scan(scan_fn, h0, (gate_z_T, z_T, gamma_T))
        
        # Transpose output back to [Batch, Seq, Dim]
        h_sequence = jnp.swapaxes(h_sequence_T, 0, 1)
        
        # 3. Post-Recurrence (Attention & FFN) - Parallel
        # Multi-Memory Attention
        # Uses HDS (functional state query)
        # CRITICAL: Pass 'q' (from input projection) as q_input for residual connection
        mem_out = self.attn(h_sequence, state, q_input=q, training=training)
        
        # Residuals
        # x_norm is original input to block (prenorm)
        # Output = x + Dropout(h_seq + mem_out + v*0.1)
        
        combined = h_sequence + mem_out + (v * 0.1)
        x_residual = x + self.dropout(combined, deterministic=not training)
        
        # FFN Execution
        x_norm2 = self.norm2(x_residual)
        
        if self.use_moe:
            # MoE Path
            ffn_out = self.moe_ffn(x_norm2, training=training)
            ffn_out = self.ffn_drop2(ffn_out, deterministic=not training)
        else:
            # Dense Path
            h_ffn = self.ffn_dense1(x_norm2)
            h_ffn = self.ffn_act(h_ffn)
            h_ffn = self.ffn_drop1(h_ffn, deterministic=not training)
            h_ffn = self.ffn_dense2(h_ffn)
            ffn_out = self.ffn_drop2(h_ffn, deterministic=not training)
        
        output = x_residual + ffn_out
        
        # 4. State Update (Functional)
        # Return new state with updated Short Term Memory
        new_state = state.update_short(k_new=h_sequence, v_new=h_sequence)
        
        # CRITICAL FIX: Update Long-Term Memory (LRU) as well
        new_state = new_state.update_long(k_new=h_sequence, v_new=h_sequence)

        # 5. UBOO Auxiliary Loss
        aux_loss = jnp.array(0.0)
        if self.use_uboo:
            # Planning Error: || W_err(stop_grad(h_t)) - W_tgt(h_{t-1}) ||^2
            # h_sequence is [Batch, Seq, Dim] (This is h_t)
            
            # Detach h_t to prevent gradient flow back through recurrence/gate from Planning Error
            h_detached = jax.lax.stop_gradient(h_sequence)
            
            # Target is h_{t-1}.
            # We need to construct h_{t-1} sequence.
            # Shift h_sequence right by 1, pad with h0 (zeros)
            # h0 is [Batch, Dim]. Expand to [Batch, 1, Dim]
            h0_expanded = jnp.zeros((batch_size, 1, self.model_dim))
            h_prev_seq = jnp.concatenate([h0_expanded, h_sequence[:, :-1, :]], axis=1)
            
            # Projections
            p_pred = self.W_planning_error(h_detached)
            p_target = self.W_planning_target(h_prev_seq)
            
            # MSE Loss
            error = p_pred - p_target
            aux_loss = jnp.mean(jnp.square(error))
        
        return output, new_state, aux_loss
