# MM-Rec Implementation Specification
## Critical Algorithmic Components

---

## 1. ASSOCIATIVE SCAN - Detailed Implementation

### 1.1 Mathematical Definition (Exponential Product)
```
Given: γ = [γ₁, γ₂, ..., γₙ] (decay coefficients)
Compute: y = [γ₁, γ₁·γ₂, γ₁·γ₂·γ₃, ..., ∏ᵢ₌₁ⁿ γᵢ]

This is a cumulative exponential product, not sum!
```

### 1.1.1 Log-Sum-Exp Implementation (CRITICAL for Stability)
```
To compute ∏ᵢ γᵢ stably:
1. Convert to log-space: log(γᵢ) for all i
2. Sum in log-space: log_sum = Σᵢ log(γᵢ)
3. Convert back: result = exp(log_sum)

Stable implementation:
  log_max = max(log(γᵢ))
  log_sum = log_max + log(Σᵢ exp(log(γᵢ) - log_max))
  result = exp(log_sum)
```

### 1.2 CUDA Kernel Structure (Exponential Product)
```cuda
// Parallel tree-based scan for cumulative exponential product
// Operates in log-space for numerical stability

__global__ void associative_scan_exponential_kernel(
    float* input,      // [batch, heads, seq_len, dim] - decay coefficients γ
    float* output,     // [batch, heads, seq_len, dim] - cumulative products
    int seq_len,
    int dim
) {
    // Step 1: Convert to log-space (with clamping)
    // log_gamma = clamp(log(gamma), -50.0f, 0.0f)
    
    // Step 2: Up-sweep phase - Build reduction tree in log-space
    // log_sum = log(gamma_i) + log(gamma_j) in log-space
    
    // Step 3: Down-sweep phase - Propagate partial results
    
    // Step 4: Convert back to linear space
    // output = exp(log_sum) with numerical safeguards
    
    // Shared memory: [blockDim.x, dim] for intermediate log states
}
```

### 1.3 PyTorch Custom Function (Exponential Product)
```python
class AssociativeScanExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma_tensor):
        """
        Compute cumulative exponential product: ∏ᵢ₌₁ᵗ γᵢ
        
        Args:
            gamma_tensor: [batch, heads, seq_len, dim] decay coefficients
        
        Returns:
            cumulative_product: [batch, heads, seq_len, dim]
        """
        # Convert to log-space with clamping
        log_gamma = torch.clamp(torch.log(gamma_tensor + 1e-8), min=-50.0, max=0.0)
        
        # Call CUDA kernel for log-space scan
        log_cumsum = associative_scan_log_kernel(log_gamma, operator='add')
        
        # Convert back to linear space with stability
        max_log = torch.max(log_cumsum, dim=-1, keepdim=True)[0]
        stable_log = log_cumsum - max_log
        cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
        
        # Save for backward
        ctx.save_for_backward(gamma_tensor, cumulative_product, log_cumsum)
        return cumulative_product
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse scan for gradient computation
        # Gradient of exp(log_sum) = exp(log_sum) * grad_output
        gamma_tensor, cumulative_product, log_cumsum = ctx.saved_tensors
        grad_gamma = cumulative_product * grad_output
        # Additional gradient computation for log-space
        return grad_gamma
```

### 1.4 Operator Requirements
- **Associativity**: (a · b) · c = a · (b · c) for multiplication
- **Log-Space Associativity**: log(a) + log(b) = log(a · b)
- **Identity Element**: 1.0 for multiplication (0.0 in log-space)
- **Numerical Stability**: MUST use Log-Sum-Exp pattern

---

## 2. HDS (HIERARCHICAL DATA STRUCTURE) - Architecture

### 2.1 Dual Memory System
```
Short-Term Memory (h_t): [batch, seq_len, hidden_dim]
    - Per-token hidden states
    - Updated via recurrence formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}

Long-Term Memory (M): [batch, M, mem_dim]
    - Persistent memory matrix (M is memory size, typically M << seq_len)
    - Access cost: O(M) - linear in memory size, not sequence length
    - Updated incrementally from h_t states
```

### 2.2 Tree Structure
```
Level 3 (Long-Term M)
    └── Level 2 (Global)
        ├── Level 1 (Block 0)      ├── Level 1 (Block 1)
        │   ├── Level 0 (h_t)      │   ├── Level 0 (h_t)
        │   ├── ...                 │   ├── ...
        │   └── Level 0 (h_t)      │   └── Level 0 (h_t)
```

### 2.2 Memory Aggregation
```python
def hds_aggregate(memories, level):
    """
    memories: [batch, seq_len, num_memories, mem_dim]
    level: hierarchy level (0=token, 1=block, 2=global)
    """
    if level == 0:
        return memories  # No aggregation
    elif level == 1:
        # Aggregate chunks of chunk_size tokens
        chunks = memories.chunk(num_chunks, dim=1)
        return torch.stack([chunk.mean(dim=1) for chunk in chunks])
    elif level == 2:
        # Global aggregation
        return memories.mean(dim=1, keepdim=True)
```

### 2.3 Hierarchical Query Mechanism
```python
def hds_query(query, memory_hierarchy):
    """
    Query at multiple levels, combine results
    """
    level_0_attn = attention(query, memory_hierarchy[0])
    level_1_attn = attention(query, memory_hierarchy[1])
    level_2_attn = attention(query, memory_hierarchy[2])
    
    # Learned combination weights
    combined = w₀ * level_0_attn + w₁ * level_1_attn + w₂ * level_2_attn
    return combined
```

### 2.4 CUDA Implementation Strategy
- **Tree Construction**: Parallel reduction tree build
- **Query Propagation**: Top-down tree traversal
- **Memory Layout**: Contiguous blocks per hierarchy level

---

## 3. CORE RECURRENCE FORMULA & MDI MECHANISM

### 3.1 Core Recurrence Formula (Efficiency Kernel)
```
h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
```

**Component Breakdown**:
- `h_t`: Current hidden state `[batch, hidden_dim]`
- `z_t`: Input/gated update `[batch, hidden_dim]` (from input projection)
- `W_g`: Gating weight matrix `[hidden_dim, hidden_dim]` (learnable)
- `σ(W_g h_{t-1})`: Gating signal (sigmoid of gated previous state)
- `γ`: Decay coefficient (scalar or per-element, learnable)
- `⊙`: Element-wise multiplication

**Parallel Computation**:
- All components computed via associative scan
- Cumulative product `∏ᵢ γᵢ` computed in log-space

### 3.2 MDI Decay Update Rule
```
memory[t] = decay_coeff[t] ⊙ memory[t-1] + (1 - decay_coeff[t]) ⊙ new_input[t]
```

Where:
- `decay_coeff[t] ∈ [0,1]` is learnable per memory bank (γ values)
- `⊙` is element-wise multiplication
- `new_input[t]` is the incoming memory update (from h_t)
- Cumulative decay: `∏ᵢ₌₁ᵗ γᵢ` via associative scan with Log-Sum-Exp

### 3.2 Learnable Decay Parameters
```python
class LearnableDecay(nn.Module):
    def __init__(self, num_memories, hidden_dim):
        self.decay_logits = nn.Parameter(
            torch.ones(num_memories) * logit(0.99)
        )
        self.decay_modulation = nn.Linear(hidden_dim, num_memories)
    
    def forward(self, memory_old, memory_new, context):
        # Base decay from parameter
        base_decay = torch.sigmoid(self.decay_logits)
        
        # Context-dependent modulation
        modulation = torch.sigmoid(self.decay_modulation(context))
        
        # Final decay coefficient
        decay = base_decay * modulation
        
        return decay * memory_old + (1 - decay) * memory_new
```

### 3.3 Integration Gate
```python
def memory_integration(old_mem, new_mem, gate):
    """
    Gated integration with residual connection
    """
    integrated = gate * new_mem + (1 - gate) * old_mem
    return integrated + old_mem  # Residual
```

### 3.4 Numerical Stability (CRITICAL)
- **Log-Sum-Exp for Exponential Products**: 
  ```python
  # Stable computation of ∏ᵢ γᵢ
  log_gamma = torch.clamp(torch.log(gamma + 1e-8), min=-50.0, max=0.0)
  log_sum = torch.cumsum(log_gamma, dim=seq_dim)
  max_log = torch.max(log_sum, dim=seq_dim, keepdim=True)[0]
  stable_log = log_sum - max_log
  result = torch.exp(stable_log) * torch.exp(max_log)
  ```
- **Clamp Decay**: Ensure `decay_coeff ∈ [ε, 1-ε]` to prevent numerical issues
- **FP32 Accumulation**: Use FP32 for decay computation, cast to FP16 for storage
- **Gradient Clipping**: Clip gradients for decay parameters separately
- **Zero/One Handling**: Special cases for γ ≈ 0 (complete decay) and γ ≈ 1 (no decay)

---

## 4. MM-REC BLOCK - Complete Flow

### 4.1 Forward Pass Pseudocode (with Core Formula)
```python
def mm_rec_block_forward(x, memory_states, long_term_memory_M):
    # 1. Input Projection
    q = linear_q(x)  # [B, T, H, D]
    k = linear_k(x)
    v = linear_v(x)
    z = linear_z(x)  # For gated update z_t
    
    # 2. Compute gating signal: σ(W_g h_{t-1})
    if memory_states is not None:
        h_prev = memory_states.hidden_states  # [B, T, D]
        gate_signal = torch.sigmoid(gating_proj(h_prev))  # [B, T, D]
    else:
        gate_signal = torch.ones_like(x) * 0.5
    
    # 3. Get decay coefficients γ
    gamma = get_decay_coefficients()  # [B, T] or [B, T, D]
    
    # 4. Core Recurrence Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    # Compute via associative scan (exponential product for γ terms)
    gamma_cumprod = associative_scan_exponential(gamma)  # ∏ᵢ γᵢ in log-space
    h_t = z * gate_signal + gamma_cumprod.unsqueeze(-1) * h_prev
    
    # 5. Multi-Memory Attention (using h_t, query against long-term M)
    mem_attn = []
    for i in range(num_memories):
        mem_k = long_term_memory_M[i]  # [B, M, D] - O(M) access
        mem_v = long_term_memory_M[i]
        attn_i = attention(h_t, mem_k, mem_v)  # Query h_t against M
        mem_attn.append(attn_i)
    mem_attn = torch.stack(mem_attn)  # [N, B, T, H, D]
    
    # 6. HDS Aggregation
    hds_levels = hds_build_hierarchy(mem_attn)
    hds_levels.append(long_term_memory_M)  # Add long-term memory level
    
    # 7. HDS Query
    hds_output = hds_query(h_t, hds_levels)
    
    # 8. MDI Update (update both h_t and M)
    updated_h_t = mdi_update(h_t, hds_output, gamma)
    updated_M = update_long_term_memory(long_term_memory_M, h_t)  # O(M) update
    
    # 9. Output Projection
    output = linear_out(torch.cat([updated_h_t, hds_output], dim=-1))
    
    return output, updated_h_t, updated_M
```

### 4.2 Memory State Structure
```python
@dataclass
class MemoryState:
    k: torch.Tensor  # [B, T, H, D]
    v: torch.Tensor  # [B, T, H, D]
    state: torch.Tensor  # [B, T, D]
    decay_coeff: torch.Tensor  # [B, T]
    hds_levels: List[torch.Tensor]  # Hierarchy levels
```

---

## 5. GRADIENT COMPUTATION DETAILS

### 5.1 Associative Scan Backward
```python
def associative_scan_backward(grad_output, intermediate_states):
    """
    Reverse scan: Start from end, propagate gradients backward
    """
    grad_input = torch.zeros_like(grad_output)
    grad_input[-1] = grad_output[-1]
    
    for t in range(seq_len - 2, -1, -1):
        # Gradient flows through scan operator
        grad_input[t] = grad_output[t] + scan_operator_grad(
            intermediate_states[t], grad_input[t+1]
        )
    
    return grad_input
```

### 5.2 HDS Gradient Flow
- **Level 2 → Level 1**: Gradient distributed to block chunks
- **Level 1 → Level 0**: Gradient distributed to individual tokens
- **Aggregation Gradients**: Mean operation → divide gradient by chunk_size

### 5.3 MDI Gradient Computation
```python
def mdi_backward(grad_new_mem, decay, old_mem, new_mem):
    grad_decay = grad_new_mem * (old_mem - new_mem)
    grad_old_mem = grad_new_mem * decay
    grad_new_mem = grad_new_mem * (1 - decay)
    return grad_old_mem, grad_new_mem, grad_decay
```

---

## 6. PERFORMANCE OPTIMIZATIONS

### 6.1 Kernel Fusion Opportunities
1. **Projection + Scan**: Fuse linear projection with scan input preparation
2. **Attention + HDS**: Combine attention computation with HDS query
3. **MDI + Normalization**: Fuse decay update with layer normalization

### 6.2 Memory Access Patterns
- **Coalesced Reads**: Ensure memory states are row-major contiguous
- **Bank Conflicts**: Avoid shared memory bank conflicts in scan kernel
- **Prefetching**: Prefetch next memory bank while processing current

### 6.3 Compute Optimization
- **Tensor Cores**: Use FP16/BF16 matrix operations where possible
- **Warp Shuffles**: Utilize `__shfl_sync()` for efficient reductions
- **Block-level Parallelism**: Process multiple memory banks in parallel

---

## 7. DISTRIBUTED TRAINING INTEGRATION

### 7.1 FSDP (Fully Sharded Data Parallel)
```python
from torch.distributed.fsdp import FullyShardedDataParallel

model = MMRecModel(config)
model = FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16
    )
)
```

### 7.2 Memory State Synchronization
- **All-Gather**: Synchronize memory states across ranks before HDS aggregation
- **Reduce-Scatter**: Distribute gradient computation for memory parameters
- **Communication Overlap**: Overlap memory sync with computation

### 7.3 Sequence Parallelism
- **Split Sequence Dimension**: Distribute sequence chunks across GPUs
- **Ring Communication**: Efficient token-by-token communication for scan
- **Gradient Accumulation**: Accumulate gradients across sequence splits

---

## 8. TESTING REQUIREMENTS

### 8.1 Numerical Correctness
```python
def test_associative_scan():
    # Compare against sequential implementation
    input_tensor = torch.randn(2, 8, 1024, 512)
    parallel_result = associative_scan(input_tensor)
    sequential_result = sequential_scan(input_tensor)
    assert torch.allclose(parallel_result, sequential_result, rtol=1e-4)
```

### 8.2 Gradient Correctness
```python
def test_gradient_flow():
    # Finite difference check
    x = torch.randn(2, 8, 1024, 512, requires_grad=True)
    y = mm_rec_block(x)
    loss = y.sum()
    loss.backward()
    
    # Compare with finite difference
    grad_autograd = x.grad
    grad_finite_diff = finite_difference(x, mm_rec_block)
    assert torch.allclose(grad_autograd, grad_finite_diff, rtol=1e-3)
```

### 8.3 Performance Benchmarks
- **Throughput**: Measure tokens/second for batch_size=1, seq_len=2048
- **Memory**: Peak GPU memory usage for 7B parameter model
- **Scaling**: Efficiency at 1, 4, 8, 16 GPUs

---

## 9. CONFIGURATION TEMPLATE

### 9.1 Model Config (7B Parameters - REQUIRED SPECS)
```python
MMREC_7B_CONFIG = {
    "vocab_size": 32000,
    "hidden_dim": 4096,          # D_hidden = 4096 (REQUIRED)
    "num_layers": 24,             # L_layer = 24 (REQUIRED)
    "num_memories": 8,
    "mem_dim": 512,
    "memory_size_M": 1024,        # Long-term memory size (M << seq_len)
    "num_heads": 32,
    "head_dim": 128,
    "ffn_dim": 11008,
    "num_hds_levels": 3,
    "chunk_size": 128,
    "max_seq_len": 32768,         # N_sequence ≥ 32768 (32K+) (REQUIRED)
    "decay_init": 0.99,           # Initial γ value
    "activation": "gelu",
    "norm_type": "rms_norm",
    "dropout": 0.1,
    "bias": False,
    "use_log_sum_exp": True,     # CRITICAL: Use Log-Sum-Exp for stability
    "log_clamp_min": -50.0,      # Clamp log(γ) to prevent underflow
    "log_clamp_max": 0.0         # Clamp log(γ) to prevent overflow
}
```

### 9.2 Training Config
```python
TRAINING_CONFIG = {
    "batch_size": 4,
    "gradient_accumulation_steps": 32,
    "effective_batch_size": 128,
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "max_steps": 100000,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "mixed_precision": "bf16",
    "compile": True,
    "gradient_checkpointing": True
}
```

---

## 10. CRITICAL IMPLEMENTATION CHECKLIST

### Phase 1: Core Components
- [ ] Associative Scan CUDA kernel (forward)
- [ ] Associative Scan backward pass
- [ ] HDS hierarchy construction
- [ ] HDS query mechanism
- [ ] MDI decay computation
- [ ] MDI integration update

### Phase 2: Block Integration
- [ ] Multi-memory attention
- [ ] Complete MM-Rec block forward
- [ ] Complete MM-Rec block backward
- [ ] Memory state management
- [ ] Gradient checkpointing hooks

### Phase 3: Optimization
- [ ] Kernel fusion (projection+scan)
- [ ] Kernel fusion (attention+HDS)
- [ ] Mixed precision support
- [ ] CUDA graph capture
- [ ] Memory access optimization

### Phase 4: Distributed Training
- [ ] FSDP integration
- [ ] Memory state synchronization
- [ ] Sequence parallelism
- [ ] Communication overlap
- [ ] Multi-node support

### Phase 5: Testing & Validation
- [ ] Unit tests for all components
- [ ] Gradient correctness tests
- [ ] Performance benchmarks
- [ ] Memory profiling
- [ ] End-to-end training run

