# MM-Rec - Complete Mechanism Deep Dive

## ✅ ALL MECHANISMS ANALYZED

---

## 1. 🔄 RECURRENCE MECHANISM (Core Formula)

### Implementation ([block.py#L159-L184](file:///home/onur/workspace/mm-rec/mm_rec_jax/blocks/block.py#L159-L184))

```python
def scan_fn(carry, inputs):
    h_prev = carry
    gate_z_t, z_t, gamma_t = inputs
    
    # 1. Gating Logic
    gate_logits = gate_z_t + self.W_g_h(h_prev)
    gate = nn.sigmoid(gate_logits)  # [0, 1]
    
    # 2. MDI Formula
    h_tilde = (1.0 - gate) * h_prev + gate * z_t
    
    # 3. Decay Integration
    h_t = h_tilde + gamma_t * h_prev
    
    # 4. Stabilization
    h_t = self.norm_recurrence(h_t)
    h_t = jnp.clip(h_t, -100.0, 100.0)
    
    return h_t, h_t
```

### How It Works:
- **Input**: `z_t` (from fused projection)
- **Gate**: Learns what to keep from prev (`h_prev`) vs new (`z_t`)
- **Decay**: `gamma` controls memory retention (0=forget, 0.9=remember)
- **Scan**: Processes full sequence in parallel via `jax.lax.scan`

### Learning:
- `W_g_z`, `W_g_h`: Learned gating weights
- Adapts based on input + previous state

---

## 2. 📉 GAMMA DECAY MECHANISM (MDI)

### Implementation ([block.py#L127-L140](file:///home/onur/workspace/mm-rec/mm_rec_jax/blocks/block.py#L127-L140))

```python
# Base Decay (from input)
gamma_hidden = nn.gelu(self.W_gamma_1(z))
gamma_base = nn.sigmoid(self.W_gamma_2(gamma_hidden))

# Context Modulation (from keys)
ctx_hidden = nn.gelu(self.W_gamma_context_1(k))
gamma_mod = nn.sigmoid(self.W_gamma_context_2(ctx_hidden))

# Combined
gamma = gamma_base * gamma_mod
gamma = jnp.clip(gamma, 1e-6, 1.0 - 1e-6)
```

### How It Works:
- **Dual pathway**: Base (from input `z`) + Modulation (from context `k`)
- **Multiply**: Combines both signals
- **Range**: [1e-6, 0.999999] → never fully 0 or 1

### Learning:
- `W_gamma_1, W_gamma_2`: Learns base decay from input
- `W_gamma_context_1, W_gamma_context_2`: Learns context-dependent modulation
- **Initialization**: -3.0 bias → starts at ~0.05 (small gamma, prevent explosion)

---

## 3. 🎯 ATTENTION MECHANISM

### Implementation ([attention.py#L25-L97](file:///home/onur/workspace/mm-rec/mm_rec_jax/blocks/attention.py#L25-L97))

```python
def __call__(self, h_t, memory_state, q_input=None, training=False):
    # 1. Project Query
    q_attention = self.q_proj(h_t)
    
    # 2. Mix with input query (residual)
    if q_input is not None:
        query = q_attention + 0.5 * q_input
    
    # 3. Get Memory via HDS
    hierarchy = HDS.construct_hierarchy(memory_state)
    k, v = HDS.query_memory(hierarchy, query, level=0)
    
    # 4. Dot Product Attention
    scores = jnp.matmul(query, k.transpose((0, 2, 1))) * self.scale
    probs = nn.softmax(scores, axis=-1)
    output = jnp.matmul(probs, v)
    
    # 5. Output Projection
    return self.out_proj(output)
```

### How It Works:
- **Standard scaled dot-product** attention
- Queries **long-term memory** via HDS
- Uses **Level 0** (full memory) by default
- O(M) complexity (M=512 slots, not O(N²))

### Learning:
- `q_proj`, `out_proj`: Learned projections
- Attention weights: Dynamically computed (softmax)

---

## 4. 🌲 HDS (Hierarchical Data Structure)

### Implementation ([hds.py#L17-L103](file:///home/onur/workspace/mm-rec/mm_rec_jax/core/hds.py#L17-L103))

```python
def construct_hierarchy(state, num_levels=3, level_ratios=(4, 4)):
    # Level 0: Raw long-term memory [Slots, Dim]
    hierarchy = {0: MemoryBank(k=current_k, v=current_v)}
    
    # Build hierarchy via average pooling
    for i in range(num_levels - 1):
        ratio = level_ratios[i]
        
        # Pool with window=ratio, stride=ratio
        current_k = nn.avg_pool(current_k, window_shape=(ratio,), strides=(ratio,))
        current_v = nn.avg_pool(current_v, window_shape=(ratio,), strides=(ratio,))
        
        hierarchy[i + 1] = MemoryBank(k=current_k, v=current_v)
    
    return hierarchy
```

### How It Works:
- **Level 0**: Full memory (512 slots)
- **Level 1**: Pooled 4:1 → 128 slots
- **Level 2**: Pooled 4:1 → 32 slots
- **Query**: Can query any level (currently uses L0)

### Purpose:
- **Multi-resolution** memory access
- **O(M)** efficient queries at different granularities
- **Currently**: Only L0 used in attention (full detail)

---

## 5. 🎲 UBOO (Unbiased Bidirectional Optimization Objective)

### Implementation ([block.py#L234-L256](file:///home/onur/workspace/mm-rec/mm_rec_jax/blocks/block.py#L234-L256))

```python
if self.use_uboo:
    # Detach h_t (no gradient flow from UBOO)
    h_detached = jax.lax.stop_gradient(h_sequence)
    
    # Target: h_{t-1} (previous states)
    h_prev_seq = jnp.concatenate([h0_expanded, h_sequence[:, :-1, :]], axis=1)
    
    # Projections
    p_pred = self.W_planning_error(h_detached)    # Predict next
    p_target = self.W_planning_target(h_prev_seq)  # From previous
    
    # Planning Error Loss
    error = p_pred - p_target
    aux_loss = jnp.mean(jnp.square(error))
```

### How It Works:
- **Predicts** next state from current (h_t)
- **Compares** to actual previous (h_{t-1})
- **Planning error**: MSE between prediction and target
- **Auxiliary loss**: Added to main loss (scaled by lambda_p=0.1)

### Purpose:
- Prevents **mode collapse**
- Encourages **temporal consistency**
- **Detached**: Doesn't interfere with main gradient flow

---

## 6. 🧠 MoE (Mixture of Experts)

### Implementation ([block.py#L27-L35](file:///home/onur/workspace/mm-rec/mm_rec_jax/blocks/block.py#L27-L35))

```python
if self.use_moe:
    self.moe_ffn = SparseMMRecBlock(
        model_dim=self.model_dim,
        num_experts=8,
        capacity_factor=1.25
    )
    # ... use in forward pass
    ffn_out = self.moe_ffn(x_norm2, training=training)
```

### How It Works:
- **Replaces** standard FFN with sparse routing
- **8 experts**: Each specializes in different patterns
- **Routing**: Top-K experts selected per token
- **Capacity**: 1.25x to handle load imbalance

### Purpose:
- **Efficiency**: Only activates subset of parameters
- **Specialization**: Different experts for different inputs
- **Scalability**: Grows model without linear compute cost

---

## 7. 🔗 MEMORY UPDATE MECHANISM

### Short-term Update ([memory_state.py#L44-L135](file:///home/onur/workspace/mm-rec/mm_rec_jax/core/memory_state.py#L44-L135))

```python
def update_short(self, k_new, v_new):
    # Ring buffer logic
    indices = (idx + jnp.arange(seq_len)) % num_slots
    k = k.at[indices].set(new_k)  # Overwrite oldest
    v = v.at[indices].set(new_v)
    new_idx = (idx + seq_len) % num_slots
    
    return self.replace(short_term=new_bank)
```

### Long-term Update ([memory_state.py#L155-L209](file:///home/onur/workspace/mm-rec/mm_rec_jax/core/memory_state.py#L155-L209))

```python
def update_long(self, k_new, v_new):
    # LRU eviction
    neg_usage = -usage_bank
    _, victim_indices = jax.lax.top_k(neg_usage, num_incoming)
    
    # Replace least-used
    k_bank = k_bank.at[victim_indices].set(k_in)
    v_bank = v_bank.at[victim_indices].set(v_in)
    usage_bank = usage_bank.at[victim_indices].set(1.0)  # Reset
```

### How It Works:
- **Short-term**: Ring buffer (FIFO, oldest dropped)
- **Long-term**: LRU cache (least-used evicted)
- **Batched**: vmap over batch dimension

---

## ✅ COMPLETE MECHANISM INVENTORY

| Mechanism | Status | Implementation | Learning |
|-----------|--------|----------------|----------|
| **Recurrence** | ✅ Full | jax.lax.scan | Gate weights |
| **Gamma Decay** | ✅ Full | Dual pathway | W_gamma × 4 |
| **Attention** | ✅ Simplified | Dot-product | q/out proj |
| **HDS Hierarchy** | ✅ Partial | 3-level pooling | None (avg pool) |
| **UBOO** | ✅ Optional | Planning error | W_error/target |
| **MoE** | ✅ Optional | Sparse routing | Router + experts |
| **Memory Update** | ✅ Full | Ring + LRU | None (algorithmic) |
| **Fused Proj** | ✅ Full | Single matmul | W_fused |
| **Cross-batch** | ✅ Implicit | State persistence | N/A |

**Research complete!** All mechanisms fully analyzed. 🎓
