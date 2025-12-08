# MM-Rec Core Recurrence Formula Specification
## Critical Algorithmic Component

---

## MATHEMATICAL DEFINITION

### Core Recurrence Formula (Efficiency Kernel)

```
h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
```

**Where:**
- `h_t`: Current hidden state `[batch, seq_len, hidden_dim]`
- `z_t`: Input/gated update `[batch, seq_len, hidden_dim]` (from input projection)
- `W_g`: Gating weight matrix `[hidden_dim, hidden_dim]` (learnable)
- `σ(W_g h_{t-1})`: Gating signal (sigmoid of gated previous state)
- `γ`: Decay coefficient (scalar or per-element, learnable)
- `⊙`: Element-wise multiplication

---

## COMPONENT BREAKDOWN

### 1. Gating Component: `z_t ⊙ σ(W_g h_{t-1})`
- **Purpose**: Selective input integration based on previous state
- **Computation**: 
  - Compute `g_t = W_g h_{t-1}` (linear projection)
  - Apply sigmoid: `σ(g_t)`
  - Element-wise multiply with input: `z_t ⊙ σ(g_t)`

### 2. Decay Component: `γ ⊙ h_{t-1}`
- **Purpose**: Exponential decay of previous state
- **Cumulative Product**: For sequence, compute `∏ᵢ₌₁ᵗ γᵢ`
- **CRITICAL**: Must use Log-Sum-Exp for numerical stability

### 3. Combined Update
- **Parallel Computation**: Both components computed via associative scan
- **Final State**: Sum of gated input and decayed previous state

---

## ASSOCIATIVE SCAN FOR EXPONENTIAL PRODUCT

### Problem
Compute cumulative exponential product: `y_t = ∏ᵢ₌₁ᵗ γᵢ`

### Solution: Log-Sum-Exp Pattern

```python
# Step 1: Convert to log-space (with clamping)
log_gamma = torch.clamp(torch.log(gamma + 1e-8), min=-50.0, max=0.0)

# Step 2: Cumulative sum in log-space
log_cumsum = torch.cumsum(log_gamma, dim=seq_dim)

# Step 3: Convert back to linear space (stable)
max_log = torch.max(log_cumsum, dim=seq_dim, keepdim=True)[0]
stable_log = log_cumsum - max_log
cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
```

### Why Log-Sum-Exp?
- **Underflow Prevention**: Direct multiplication of many small γ values → 0
- **Overflow Prevention**: Direct multiplication of many large γ values → Inf
- **Numerical Stability**: Log-space operations are more stable
- **Precision**: Maintains numerical precision for long sequences

---

## IMPLEMENTATION REQUIREMENTS

### 1. Gating Projection
```python
self.gating_proj = nn.Linear(hidden_dim, hidden_dim)
gate_signal = torch.sigmoid(self.gating_proj(h_prev))
```

### 2. Input Projection for z_t
```python
self.z_proj = nn.Linear(hidden_dim, hidden_dim)
z_t = self.z_proj(x)
```

### 3. Decay Coefficients
```python
# Learnable decay parameter
self.decay_logits = nn.Parameter(torch.ones(num_memories) * logit(0.99))
gamma = torch.sigmoid(self.decay_logits)
```

### 4. Cumulative Product (via Associative Scan)
```python
from ..core.associative_scan import associative_scan_exponential
gamma_cumprod = associative_scan_exponential(gamma)
```

### 5. Final Computation
```python
h_t = z_t * gate_signal + gamma_cumprod * h_prev
```

---

## NUMERICAL STABILITY GUARANTEES

### 1. Log-Space Clamping
- **Min**: `-50.0` (prevents exp(-50) ≈ 0 underflow)
- **Max**: `0.0` (prevents exp(0) = 1 overflow)
- **Epsilon**: `1e-8` added before log to prevent log(0)

### 2. Stable Exponential Computation
```python
# Pattern: exp(log_sum - max) * exp(max)
max_log = torch.max(log_cumsum)
stable_exp = torch.exp(log_cumsum - max_log) * torch.exp(max_log)
```

### 3. Special Cases
- **γ ≈ 0**: Complete decay → `h_t ≈ z_t ⊙ σ(W_g h_{t-1})`
- **γ ≈ 1**: No decay → `h_t ≈ z_t ⊙ σ(W_g h_{t-1}) + h_{t-1}`
- **γ = 1**: Identity → `h_t = z_t ⊙ σ(W_g h_{t-1}) + h_{t-1}`

---

## GRADIENT COMPUTATION

### Forward Pass
```python
h_t = z_t * gate_signal + gamma_cumprod * h_prev
```

### Backward Pass
```python
# Gradient w.r.t. z_t
grad_z_t = grad_h_t * gate_signal

# Gradient w.r.t. gate_signal
grad_gate = grad_h_t * z_t
grad_gating_proj = grad_gate * gate_signal * (1 - gate_signal) @ h_prev

# Gradient w.r.t. gamma_cumprod
grad_gamma_cumprod = grad_h_t * h_prev

# Gradient w.r.t. gamma (via associative scan backward)
grad_gamma = associative_scan_exponential_backward(grad_gamma_cumprod, gamma_cumprod)

# Gradient w.r.t. h_prev
grad_h_prev = grad_h_t * gamma_cumprod
```

---

## PARALLEL COMPUTATION

### Associative Scan Benefits
- **O(log n) Depth**: Parallel tree-based computation
- **Scalable**: Works for sequences of length 32K+
- **Efficient**: Single pass through sequence

### CUDA Implementation
- **Log-Space Operations**: All intermediate computations in log-space
- **Warp-Level Primitives**: Efficient log-space reductions
- **Memory Coalescing**: Optimized access patterns

---

## CONFIGURATION

### Required Parameters
```python
{
    "hidden_dim": 4096,          # D_hidden (REQUIRED)
    "num_layers": 24,             # L_layer (REQUIRED)
    "max_seq_len": 32768,         # N_sequence ≥ 32K (REQUIRED)
    "decay_init": 0.99,           # Initial γ value
    "use_log_sum_exp": True,       # CRITICAL: Use Log-Sum-Exp
    "log_clamp_min": -50.0,       # Prevent underflow
    "log_clamp_max": 0.0          # Prevent overflow
}
```

---

## TESTING REQUIREMENTS

### 1. Correctness Test
```python
# Compare against sequential implementation
h_t_seq = sequential_recurrence(z_t, h_prev, gamma, W_g)
h_t_parallel = parallel_recurrence(z_t, h_prev, gamma, W_g)
assert torch.allclose(h_t_seq, h_t_parallel, rtol=1e-4)
```

### 2. Numerical Stability Test
```python
# Test with extreme γ values
gamma_small = torch.ones(1000) * 0.001  # Many small values
gamma_large = torch.ones(1000) * 0.999  # Many large values

# Should not produce NaN/Inf
result_small = associative_scan_exponential(gamma_small)
result_large = associative_scan_exponential(gamma_large)

assert not torch.isnan(result_small).any()
assert not torch.isinf(result_small).any()
assert not torch.isnan(result_large).any()
assert not torch.isinf(result_large).any()
```

### 3. Gradient Test
```python
# Finite difference validation
x = torch.randn(2, 2048, 4096, requires_grad=True)
y = core_recurrence_formula(x)
loss = y.sum()
loss.backward()

grad_autograd = x.grad
grad_finite_diff = finite_difference(x, core_recurrence_formula)
assert torch.allclose(grad_autograd, grad_finite_diff, rtol=1e-3)
```

---

## CRITICAL NOTES

1. **Log-Sum-Exp is MANDATORY**: Direct multiplication will fail for long sequences
2. **Clamping is REQUIRED**: Prevents numerical underflow/overflow
3. **FP32 for Log Operations**: Use FP32 for log-space computations
4. **Gradient Checkpointing**: May be needed for very long sequences
5. **Memory Efficiency**: O(M) memory access, not O(N) where N is sequence length

