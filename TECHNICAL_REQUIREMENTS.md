# MM-Rec Architecture: PyTorch/JAX Implementation Requirements
## Technical Engineering Outputs for 7B+ Parameter Training

---

## 1. CORE DATA STRUCTURES & TENSOR LAYOUTS

### 1.1 Memory State Tensors
- **Multi-Memory Buffer**: `[batch, seq_len, num_memories, mem_dim]` layout specification
- **Memory Decay Coefficients**: `[batch, seq_len, num_memories]` float32 tensor
- **Associative State**: `[batch, num_heads, mem_dim, mem_dim]` for parallel scan operations
- **HDS Hierarchy Levels**: Nested tensor structure for hierarchical memory access patterns

### 1.2 Input/Output Specifications
- **Token Embeddings**: `[batch, seq_len, hidden_dim]` → Memory projection matrices
- **Memory Query/Key/Value**: Separate projection paths for each memory bank
- **Output Projection**: `[batch, seq_len, num_memories, mem_dim]` → `[batch, seq_len, hidden_dim]`

---

## 2. ASSOCIATIVE SCAN IMPLEMENTATION

### 2.1 Core Algorithm: Parallel Cumulative Exponential Product
- **Operator**: Exponential multiplication (not addition) - `∏ᵢ γᵢ`
- **Mathematical Form**: Cumulative product of decay coefficients: `y_t = ∏ᵢ₌₁ᵗ γᵢ`
- **Numerical Stability**: **CRITICAL** - Must use Log-Sum-Exp trick to prevent underflow/overflow
- **Log-Sum-Exp Implementation**: `log(∏γ) = Σ log(γ)`, then `exp(Σ log(γ))` with numerical safeguards

### 2.2 CUDA Kernel Requirements
- **Parallel Prefix Product Kernel**: Block-level associative scan for O(log n) depth
- **Log-Space Computation**: Operate in log-space to prevent numerical issues
- **Warp-level Primitives**: `__shfl_sync()` operations for intra-warp reductions
- **Memory Coalescing**: Stride patterns for `[batch, num_heads, seq_len, mem_dim]` access
- **Shared Memory Tiling**: 32KB shared memory blocks for intermediate scan states
- **Numerical Safeguards**: Clamp log values, handle zero/negative inputs

### 2.3 PyTorch Custom Op
- **torch.autograd.Function**: Forward/backward pass for associative scan
- **Log-Space Forward**: Compute in log-space, convert back to linear space at end
- **Gradient Checkpointing**: Selective recomputation points in scan tree
- **Mixed Precision**: FP16/BF16 support with loss scaling for numerical stability
- **Stability Checks**: Runtime validation of log-space intermediate values

### 2.3 JAX Implementation
- **jax.lax.associative_scan**: Native JAX primitive utilization
- **vmap/pmap**: Batch and model parallelism integration
- **Custom XLA HLO**: Lower-level optimization for TPU/GPU backends

---

## 3. HDS (HIERARCHICAL DATA STRUCTURE) ENGINE

### 3.1 Dual Memory System Architecture
- **Short-Term Memory (h_t)**: Per-token hidden states `[batch, seq_len, hidden_dim]`
- **Long-Term Memory (M)**: Persistent memory matrix `[batch, M, mem_dim]` where M is memory size
- **Memory Access Cost**: O(M) - linear in memory size, not sequence length
- **Memory Update**: Incremental updates to M based on h_t states

### 3.2 Multi-Level Memory Hierarchy
- **Level 0**: Per-token local memory (h_t) - Short-term
- **Level 1**: Block-level aggregated memory (chunk_size tokens)
- **Level 2**: Sequence-level global memory
- **Level 3**: Long-term memory M - Persistent across sequences

### 3.3 Hierarchical Access Patterns
- **Bottom-Up Aggregation**: Tree reduction kernel for memory consolidation
- **Top-Down Query**: Hierarchical attention mechanism across levels
- **Dynamic Level Selection**: Learned routing for memory access decisions
- **Memory Retrieval**: O(M) cost for querying long-term memory M

### 3.3 CUDA Implementation Details
- **Tree Traversal Kernels**: Parallel tree walk for O(log n) access
- **Memory Pool Management**: Pre-allocated buffers for each hierarchy level
- **Atomic Operations**: Lock-free updates for concurrent memory writes

---

## 4. CORE RECURRENCE FORMULA & MDI MECHANISM

### 4.1 Core Recurrence Formula (Efficiency Kernel)
**Mathematical Definition**:
```
h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
```

Where:
- `h_t`: Current hidden state `[batch, hidden_dim]`
- `z_t`: Input/gated update `[batch, hidden_dim]`
- `W_g`: Gating weight matrix `[hidden_dim, hidden_dim]`
- `σ`: Sigmoid activation
- `γ`: Decay coefficient (scalar or per-element)
- `⊙`: Element-wise multiplication

### 4.2 Implementation Requirements
- **Gating Mechanism**: `σ(W_g h_{t-1})` - Learnable gating based on previous state
- **Decay Component**: `γ ⊙ h_{t-1}` - Exponential decay of previous state
- **Update Component**: `z_t ⊙ σ(W_g h_{t-1})` - Gated input integration
- **Parallel Computation**: All components computed in parallel via associative scan

### 4.3 Memory Decay/Integration (MDI)
- **Exponential Decay**: `memory[t] = memory[t-1] * decay_coeff + new_input`
- **Learnable Decay Rates**: Per-memory-bank decay parameters (γ values)
- **Time-Varying Decay**: Adaptive decay based on sequence position
- **Cumulative Product**: `∏ᵢ γᵢ` computed via associative scan with Log-Sum-Exp

### 4.4 Integration Logic
- **Gated Memory Update**: Sigmoid gates for selective memory integration
- **Residual Connections**: Skip connections for gradient flow
- **Memory Normalization**: LayerNorm/RMSNorm for memory state stability

### 4.5 Numerical Stability (CRITICAL)
- **Log-Sum-Exp for Exponential Products**: 
  - Compute `log(∏ᵢ γᵢ) = Σᵢ log(γᵢ)` in log-space
  - Use stable exp computation: `exp(log_sum - max_log)` pattern
  - Prevent underflow: clamp `log(γ)` to `[-50, 0]` range
  - Prevent overflow: subtract max before exp, add back after
- **FP32 Accumulation**: High-precision accumulation for decay operations
- **Gradient Clipping**: Per-memory-bank gradient norm constraints
- **NaN/Inf Detection**: Runtime checks for numerical anomalies
- **Zero Handling**: Special case for γ ≈ 0 (complete decay)
- **One Handling**: Special case for γ ≈ 1 (no decay)

---

## 5. MM-REC BLOCK ARCHITECTURE

### 5.1 Block Components (Sequential Order)
1. **Input Projection**: Linear layers for memory query/key/value
2. **Multi-Memory Attention**: Parallel attention over N memory banks
3. **Associative Scan**: Parallel prefix computation over sequence
4. **HDS Aggregation**: Hierarchical memory consolidation
5. **MDI Update**: Memory decay and integration step
6. **Output Projection**: Memory states → hidden representations

### 5.2 Memory Bank Specifications
- **Number of Banks**: Configurable (default: 4-8 for 7B model)
- **Bank Dimensions**: `mem_dim = hidden_dim // num_banks`
- **Bank Specialization**: Optional learned routing for task-specific memories

### 5.3 Activation Functions
- **Memory Activations**: GELU/SiLU for memory state transformations
- **Gate Activations**: Sigmoid for gating mechanisms
- **Output Activations**: GELU for final hidden states

---

## 6. DISTRIBUTED TRAINING REQUIREMENTS

### 6.1 Model Parallelism
- **Memory Bank Sharding**: Distribute memory banks across GPUs
- **Sequence Parallelism**: Split sequence dimension for long contexts
- **Pipeline Parallelism**: Layer-wise partitioning for deep models

### 6.2 Communication Patterns
- **All-Gather**: Memory state synchronization across devices
- **All-Reduce**: Gradient aggregation for memory parameters
- **Ring Attention**: Efficient communication for associative scan

### 6.3 Memory Efficiency
- **Gradient Checkpointing**: Trade compute for memory in backward pass
- **CPU Offloading**: Offload inactive memory banks to CPU RAM
- **Activation Recomputation**: Selective recomputation for memory states

---

## 7. OPTIMIZATION & PERFORMANCE

### 7.1 Kernel Fusion
- **Fused Memory Operations**: Combine projection + scan + update
- **Fused Attention**: Memory attention + HDS query in single kernel
- **Fused Activation**: GELU + normalization in one pass

### 7.2 Memory Access Optimization
- **Tensor Contiguity**: Ensure row-major layout for coalesced access
- **Prefetching**: Prefetch next memory bank while processing current
- **Cache Blocking**: Tile memory operations to fit in L2 cache

### 7.3 Compilation Targets
- **PyTorch**: `torch.compile()` with `mode="reduce-overhead"`
- **JAX**: `jax.jit()` with `donate_argnums` for memory efficiency
- **CUDA Graphs**: Capture and replay kernel sequences for reduced launch overhead

---

## 8. GRADIENT COMPUTATION & AUTODIFF

### 8.1 Custom Backward Pass
- **Associative Scan Gradients**: Reverse-mode AD for scan operations
- **HDS Gradient Flow**: Backprop through hierarchical structure
- **MDI Gradient Accumulation**: Handle decay parameter gradients

### 8.2 Numerical Gradients
- **Finite Difference Checks**: Validate custom backward implementations
- **Gradient Norm Monitoring**: Track gradient magnitudes per component
- **Exploding Gradient Detection**: Automatic gradient clipping triggers

---

## 9. INITIALIZATION & STABILITY

### 9.1 Weight Initialization
- **Memory Bank Init**: Xavier/Kaiming for projection matrices
- **Decay Parameter Init**: Start near 1.0 (minimal decay) with small variance
- **HDS Level Init**: Zero-initialize aggregation weights

### 9.2 Training Stability
- **Learning Rate Schedule**: Warmup + cosine decay for memory parameters
- **Layer-wise LR**: Different rates for memory vs. projection layers
- **EMA for Memory States**: Exponential moving average for inference

---

## 10. PROFILING & DEBUGGING TOOLS

### 10.1 Performance Metrics
- **Kernel Timing**: CUDA events for each custom operation
- **Memory Bandwidth**: Track memory throughput vs. compute utilization
- **FLOP Counting**: Accurate FLOP estimates for MM-Rec operations

### 10.2 Debugging Utilities
- **Memory State Visualization**: Tensor inspection tools for memory banks
- **Gradient Flow Tracking**: Identify vanishing/exploding gradient sources
- **Numerical Stability Checks**: Automatic detection of NaN/Inf propagation

---

## 11. TESTING & VALIDATION

### 11.1 Unit Tests
- **Associative Scan Correctness**: Compare against sequential implementation
- **HDS Hierarchy Integrity**: Verify memory aggregation at each level
- **MDI Decay Accuracy**: Validate decay computation precision

### 11.2 Integration Tests
- **End-to-End Forward Pass**: Full block forward with random inputs
- **Gradient Check**: Finite difference vs. autograd comparison
- **Multi-GPU Correctness**: Verify distributed training equivalence

### 11.3 Benchmarking
- **Throughput**: Tokens/second for various sequence lengths
- **Memory Usage**: Peak GPU memory for 7B parameter model
- **Scaling Efficiency**: Strong/weak scaling across GPU counts

---

## 12. CONFIGURATION & HYPERPARAMETERS

### 12.1 Architecture Config (7B Model)
```python
{
    "hidden_dim": 4096,          # D_hidden = 4096 (REQUIRED)
    "num_layers": 24,             # L_layer = 24 (REQUIRED)
    "num_memories": 8,
    "mem_dim": 512,
    "num_hds_levels": 3,
    "chunk_size": 128,            # For HDS Level 1
    "decay_init": 0.99,           # Initial γ value
    "num_heads": 32,
    "ffn_dim": 11008,
    "max_seq_len": 32768,         # N_sequence ≥ 32768 (32K+) (REQUIRED)
    "vocab_size": 32000
}
```

### 12.2 Training Config
- **Batch Size**: Gradient accumulation for effective large batches
- **Sequence Length**: **MUST support 32K+ tokens** (32768+) with efficient memory
- **Mixed Precision**: BF16 for training, FP16 for inference
- **Memory Efficiency**: O(M) memory access cost, not O(N) where N is sequence length

---

## 13. DEPLOYMENT CONSIDERATIONS

### 13.1 Inference Optimization
- **KV Cache Equivalent**: Persistent memory state caching
- **Incremental Decoding**: Update memory states token-by-token
- **Quantization**: INT8/INT4 support for memory banks

### 13.2 Model Serialization
- **Checkpoint Format**: Separate storage for memory states vs. weights
- **State Dict Compatibility**: PyTorch standard format with custom extensions
- **Versioning**: Architecture version tracking for compatibility

---

## DELIVERABLES CHECKLIST

- [ ] Associative Scan CUDA kernel (forward + backward)
- [ ] HDS hierarchy implementation with tree traversal
- [ ] MDI decay mechanism with learnable parameters
- [ ] Complete MM-Rec block in PyTorch/JAX
- [ ] Distributed training support (FSDP/DDP)
- [ ] Mixed precision training pipeline
- [ ] Gradient checkpointing integration
- [ ] Performance profiling tools
- [ ] Unit test suite
- [ ] 7B parameter model configuration
- [ ] Training script with hyperparameter tuning
- [ ] Inference optimization (incremental decoding)

