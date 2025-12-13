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
        self.W_gamma_1 = nn.Dense(self.model_dim // 4) # Inner dim default is dim // 4
        self.W_gamma_2 = nn.Dense(self.model_dim)
        
        # Context Modulation (MDI)
        # Modulates gamma based on context (Keys)
        self.W_gamma_context_1 = nn.Dense(self.model_dim // 4)
        self.W_gamma_context_2 = nn.Dense(self.model_dim)
        
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
        self.ffn_dense1 = nn.Dense(self.ffn_dim)
        self.ffn_act = nn.gelu
        self.ffn_drop1 = nn.Dropout(self.dropout_rate)
        self.ffn_dense2 = nn.Dense(self.model_dim)
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
        
        # 2. Recurrence Scan Function
        # Scan carries: (h_prev)
        # Inputs: (q_t, k_t, v_t, z_t, gamma_t)
        
        def scan_fn(carry, inputs):
            h_prev = carry
            z_t, gamma_t = inputs
            
            # Gating Logic (MDI style)
            # gate = Sigmoid(W_g([z_t, h_prev]))
            # In PyTorch MDI, W_g takes concat(z_t, h_prev).
            # My self.W_g here was initialized as Dense(model_dim). 
            # PyTorch MDI has Linear(2*dim, dim).
            # I need to concat and use correct weight (or separate weights if I want to avoid concat overhead).
            
            # But wait, self.W_g in my JAX setup was Dense(model_dim).
            # I should update setup() too to be Dense(model_dim) but taking 2x input implicitly?
            # Or just separate: gate = sigmoid(W_g1(z) + W_g2(h))
            
            # Let's check PyTorch again: `self.W_g = nn.Linear(model_dim * 2, model_dim)`
            # So it does concat.
            
            # To match EXACTLY:
            # I will use two Dense layers in setup: W_g_z and W_g_h and sum them.
            # This is mathematically equivalent to W_g(concat(z, h)).
            
            gate_logits = self.W_g_z(z_t) + self.W_g_h(h_prev)
            gate = nn.sigmoid(gate_logits)
            
            # MDI Formula:
            # h_tilde = (1 - gate) * h_prev + gate * z_t
            h_tilde = (1.0 - gate) * h_prev + gate * z_t
            
            # Final: h_new = h_tilde + gamma * h_prev
            h_t = h_tilde + gamma_t * h_prev
            
            return h_t, h_t

        # Initial hidden state (zeros) [Batch, Dim]
        h0 = jnp.zeros((batch_size, self.model_dim))
        
        # Run Scan
        # We scan over axis 1 (Time)
        # scan arguments: (f, init, xs)
        # xs structure must match inputs of scan_fn
        xs = (z, gamma)
        # We need to transpose to [Seq, Batch, Dim] for scan iteration usually,
        # but flax.linen.scan handles leading axis scan if configured.
        # But jax.lax.scan expects leading axis scan.
        
        # Transpose inputs to [Seq, Batch, Dim]
        z_T = jnp.swapaxes(z, 0, 1)
        gamma_T = jnp.swapaxes(gamma, 0, 1)
        
        # Run scan
        _, h_sequence_T = jax.lax.scan(scan_fn, h0, (z_T, gamma_T))
        
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
