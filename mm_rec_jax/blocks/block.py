import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any

from ..core.memory_state import MemoryState
from ..core.hds import HDS
from .attention import MultiMemoryAttention

class MMRecBlock(nn.Module):
    """
    MM-Rec Block (JAX/Flax).
    Implements efficient gated recurrence using jax.lax.scan.
    """
    model_dim: int
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout_rate: float = 0.1
    
    def setup(self):
        # Projections
        # Using Xavier Uniform to match PyTorch
        self.W_q = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        self.W_k = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        self.W_v = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        self.W_z = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        
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
        
        # FFN (Explicit definition to handle 'deterministic' arg correctly)
        self.ffn_dense1 = nn.Dense(self.ffn_dim, kernel_init=nn.initializers.xavier_uniform())
        self.ffn_act = nn.gelu
        self.ffn_drop1 = nn.Dropout(self.dropout_rate)
        self.ffn_dense2 = nn.Dense(self.model_dim, kernel_init=nn.initializers.xavier_uniform())
        self.ffn_drop2 = nn.Dropout(self.dropout_rate)
        
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, 
                 x: jnp.ndarray, 
                 state: MemoryState, 
                 training: bool = False) -> Tuple[jnp.ndarray, MemoryState]:
        """
        Forward pass with Scan.
        
        Args:
            x: Input [Batch, Seq, Dim]
            state: Functional MemoryState
            
        Returns:
            (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections (Parallel)
        x_norm = self.norm1(x)
        q = self.W_q(x_norm)
        k = self.W_k(x_norm)
        v = self.W_v(x_norm)
        z = self.W_z(x_norm)
        
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
        
        # FFN
        x_norm2 = self.norm2(x_residual)
        
        # Manual FFN execution
        h_ffn = self.ffn_dense1(x_norm2)
        h_ffn = self.ffn_act(h_ffn)
        h_ffn = self.ffn_drop1(h_ffn, deterministic=not training)
        h_ffn = self.ffn_dense2(h_ffn)
        ffn_out = self.ffn_drop2(h_ffn, deterministic=not training)
        
        output = x_residual + ffn_out
        
        # 4. State Update (Functional)
        # Return new state with updated Short Term Memory
        # We update with the calculated h_sequence (Keys/Values for next blocks)
        new_state = state.update_short(k_new=h_sequence, v_new=h_sequence)
        
        return output, new_state
