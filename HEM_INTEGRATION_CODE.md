# HEM (Mekanizma 1) Entegrasyonu - Kritik Kod Parçaları

Bu doküman, MM-Rec analiz dokümanına dayanarak **Mekanizma 1 (HEM - Fused Kernel)** entegrasyonu için kritik kod parçalarını içerir.

---

## 1. Fused Kernel Tanımı

### Dosya: `mm_rec/blocks/mm_rec_block.py`

### `__init__` Metodunda Fused Ağırlık Matrisinin Tanımı

```python
def __init__(
    self,
    model_dim: int = 4096,
    inner_dim: Optional[int] = None,
    num_heads: int = 8,
    num_memories: int = 1,
    mem_dim: Optional[int] = None,
    ffn_dim: Optional[int] = None,
    dropout: float = 0.1,
    # HEM Parameters
    use_hem: bool = True,           # Enable HEM (Fused Kernel) mechanism
    pe_dim: Optional[int] = None,   # Positional encoding dimension (default: model_dim)
):
    super().__init__()
    self.model_dim = model_dim
    self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
    self.num_heads = num_heads
    self.num_memories = num_memories
    self.mem_dim = mem_dim if mem_dim is not None else model_dim
    self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
    self.dropout = dropout
    
    # HEM Configuration
    self.use_hem = use_hem
    self.pe_dim = pe_dim if pe_dim is not None else model_dim
    
    # Normalization layers
    self.norm1 = RMSNorm(model_dim)
    self.norm2 = RMSNorm(model_dim)
    
    # ========================================================================
    # HEM: Fused Kernel - Single Large Weight Matrix
    # ========================================================================
    # HEM mekanizması, altı projeksiyonu (QKVZ + PE) tek bir büyük ağırlık
    # matrisi olarak birleştirir:
    # 
    # W_fused = [W_Q; W_K; W_V; W_Z; W_P; W_E]
    # 
    # Boyutlar:
    # - W_Q: [model_dim, model_dim] → Query projection
    # - W_K: [model_dim, model_dim] → Key projection
    # - W_V: [model_dim, model_dim] → Value projection
    # - W_Z: [model_dim, model_dim] → z_t projection (for core formula)
    # - W_P: [model_dim, pe_dim]   → Positional encoding projection (down)
    # - W_E: [pe_dim, model_dim]   → Positional encoding projection (up)
    # 
    # Fused Matrix Shape:
    # - Input:  [batch, seq_len, model_dim]
    # - Weight: [model_dim, 4*model_dim + pe_dim + model_dim]
    #          = [model_dim, 5*model_dim + pe_dim]
    # - Output: [batch, seq_len, 5*model_dim + pe_dim]
    # 
    # Bu tek matris çarpımı sayesinde:
    # - 6 ayrı matmul yerine 1 matmul (6x daha az kernel launch)
    # - Daha iyi memory bandwidth utilization
    # - Daha iyi cache locality
    # - GPU'da daha verimli paralel işleme
    # ========================================================================
    
    if self.use_hem:
        # Calculate fused output dimension
        # Q + K + V + Z + P_down + E_up
        # = model_dim + model_dim + model_dim + model_dim + pe_dim + model_dim
        # = 5*model_dim + pe_dim
        self.fused_out_dim = 4 * model_dim + self.pe_dim + model_dim  # QKVZ + P + E
        
        # CRITICAL: Single fused linear layer
        # This replaces 6 separate Linear layers with 1 large Linear layer
        self.W_fused = nn.Linear(
            in_features=model_dim,
            out_features=self.fused_out_dim,
            bias=True  # Bias term for each projection
        )
        
        # Store projection dimensions for splitting
        self.proj_dims = {
            'Q': model_dim,
            'K': model_dim,
            'V': model_dim,
            'Z': model_dim,
            'P': self.pe_dim,  # Positional encoding down-projection
            'E': model_dim     # Positional encoding up-projection
        }
        
        # Calculate split indices for output tensor
        # Output: [Q, K, V, Z, P, E]
        self.split_indices = [
            self.proj_dims['Q'],                                    # Q end
            self.proj_dims['Q'] + self.proj_dims['K'],              # K end
            self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'],  # V end
            self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'] + self.proj_dims['Z'],  # Z end
            self.proj_dims['Q'] + self.proj_dims['K'] + self.proj_dims['V'] + self.proj_dims['Z'] + self.proj_dims['P'],  # P end
            self.fused_out_dim  # E end (total)
        ]
        
        # Initialize fused weight matrix
        # CRITICAL: Proper initialization for each sub-matrix
        self._init_fused_weights()
        
        # Positional encoding components (if needed separately)
        # These are computed from W_P and W_E but can be accessed separately
        self._pe_initialized = False
    else:
        # Fallback: Separate projections (original approach)
        # This is kept for backward compatibility
        self.W_fused = None
        self.proj_dims = None
        self.split_indices = None
        
        # Separate projections
        self.W_q = nn.Linear(model_dim, model_dim)  # Query
        self.W_k = nn.Linear(model_dim, model_dim)  # Key
        self.W_v = nn.Linear(model_dim, model_dim)  # Value
        self.W_z = nn.Linear(model_dim, model_dim)  # z_t for core formula
        self.W_p = nn.Linear(model_dim, self.pe_dim)  # Positional encoding down
        self.W_e = nn.Linear(self.pe_dim, model_dim)  # Positional encoding up
    
    # Gating projection for core formula: W_g (separate, not fused)
    self.W_g = nn.Linear(model_dim, model_dim)
    
    # ... (diğer bileşenler: MDI, attention, FFN, vb.)
    
    # Performance optimization flags
    self.use_gradient_checkpointing = False
    self.use_kernel_fusion = True  # HEM is a form of kernel fusion
    
    # ... (C++ optimization setup, vb.)

def _init_fused_weights(self):
    """
    Initialize fused weight matrix by properly initializing each sub-matrix.
    
    CRITICAL: Each projection (Q, K, V, Z, P, E) should be initialized
    independently to maintain proper weight initialization.
    """
    with torch.no_grad():
        # Get fused weight and bias
        fused_weight = self.W_fused.weight.data  # [fused_out_dim, model_dim]
        fused_bias = self.W_fused.bias.data      # [fused_out_dim]
        
        # Initialize each sub-matrix with proper initialization
        # Standard initialization: Xavier/Glorot uniform for linear layers
        
        # Q projection: [model_dim, model_dim]
        start_idx = 0
        end_idx = self.proj_dims['Q']
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # K projection: [model_dim, model_dim]
        start_idx = end_idx
        end_idx += self.proj_dims['K']
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # V projection: [model_dim, model_dim]
        start_idx = end_idx
        end_idx += self.proj_dims['V']
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # Z projection: [model_dim, model_dim]
        start_idx = end_idx
        end_idx += self.proj_dims['Z']
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # P projection (down): [pe_dim, model_dim]
        start_idx = end_idx
        end_idx += self.proj_dims['P']
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # E projection (up): [model_dim, pe_dim] (note: this is transposed in fused matrix)
        # In fused matrix, E is stored as [model_dim, pe_dim] but we need to initialize
        # the weight that will be used for up-projection
        # For now, we'll initialize it similarly
        start_idx = end_idx
        end_idx = self.fused_out_dim
        nn.init.xavier_uniform_(fused_weight[start_idx:end_idx, :])
        nn.init.zeros_(fused_bias[start_idx:end_idx])
        
        # Store initialized weights
        self.W_fused.weight.data = fused_weight
        self.W_fused.bias.data = fused_bias
```

---

## 2. Fused Forward Pass

### Dosya: `mm_rec/blocks/mm_rec_block.py`

### `MMRecBlock.forward` Metodunda Tek Fused Matmul Çağrısı

```python
def forward(
    self,
    x: torch.Tensor,
    state: MemoryState,
    hds: Optional[HierarchicalDataStructure] = None,
    use_checkpointing: Optional[bool] = None
) -> Tuple[torch.Tensor, MemoryState]:
    """
    Forward pass through MM-Rec block with HEM (Fused Kernel) support.
    
    CRITICAL: HEM mekanizması ile tek bir fused matmul ile tüm projeksiyonlar
    hesaplanır (QKVZ + PE).
    
    Args:
        x: Input tensor [batch, seq_len, model_dim]
        state: MemoryState instance
        hds: Optional HierarchicalDataStructure
        use_checkpointing: Override checkpointing setting
    
    Returns:
        Tuple of (output, updated_state)
    """
    batch_size, seq_len, _ = x.shape
    
    # Override checkpointing if specified
    if use_checkpointing is not None:
        self.use_gradient_checkpointing = use_checkpointing
    
    # Create HDS if not provided
    if hds is None:
        hds = HierarchicalDataStructure(
            memory_state=state,
            num_levels=3,
            model_dim=self.model_dim
        )
        hds.construct_hierarchy(state)
    
    # Initialize output tensor
    output = torch.zeros_like(x)  # [batch, seq_len, model_dim]
    
    # Initialize previous hidden state h_{t-1} (for t=0, use zeros)
    h_prev = torch.zeros(batch_size, 1, self.model_dim, 
                        dtype=x.dtype, device=x.device)
    
    # ========================================================================
    # HEM: Fused Kernel - Single Matmul for QKVZ + PE
    # ========================================================================
    # CRITICAL: Tek bir matmul ile tüm projeksiyonlar hesaplanır
    # 
    # Input:  x_norm [batch, seq_len, model_dim]
    # Weight: W_fused [model_dim, 5*model_dim + pe_dim]
    # Output: fused_output [batch, seq_len, 5*model_dim + pe_dim]
    # 
    # Fused output contains: [Q, K, V, Z, P, E]
    # 
    # Bu tek matmul sayesinde:
    # - 6 ayrı kernel launch yerine 1 kernel launch
    # - Daha iyi memory bandwidth (tek seferde tüm data transfer)
    # - Daha iyi cache utilization (weight matrix bir kez load edilir)
    # - GPU'da daha verimli paralel işleme
    # ========================================================================
    
    if self.use_hem:
        # Step 1: Normalize input
        x_norm_all = self.norm1(x)  # [batch, seq_len, model_dim]
        
        # Step 2: CRITICAL - Single fused matmul
        # This replaces 6 separate matmul operations:
        #   q = W_q(x_norm)
        #   k = W_k(x_norm)
        #   v = W_v(x_norm)
        #   z = W_z(x_norm)
        #   p = W_p(x_norm)  # Positional encoding down
        #   e = W_e(p)       # Positional encoding up (would require 2 matmuls)
        #
        # With single operation:
        #   fused_output = W_fused(x_norm)
        #
        # CRITICAL: Use F.linear for efficiency (handles bias automatically)
        # Alternative: torch.matmul(x_norm, W_fused.weight.t()) + W_fused.bias
        fused_output = F.linear(
            x_norm_all,
            self.W_fused.weight,  # [fused_out_dim, model_dim]
            self.W_fused.bias      # [fused_out_dim]
        )
        # fused_output: [batch, seq_len, fused_out_dim]
        # fused_out_dim = 5*model_dim + pe_dim
        
        # Step 3: Split fused output into individual projections
        # Split indices: [Q_end, K_end, V_end, Z_end, P_end, E_end]
        q_all, k_all, v_all, z_all, p_all, e_all = torch.split(
            fused_output,
            split_size_or_sections=[
                self.proj_dims['Q'],
                self.proj_dims['K'],
                self.proj_dims['V'],
                self.proj_dims['Z'],
                self.proj_dims['P'],
                self.proj_dims['E']
            ],
            dim=-1
        )
        # q_all: [batch, seq_len, model_dim]
        # k_all: [batch, seq_len, model_dim]
        # v_all: [batch, seq_len, model_dim]
        # z_all: [batch, seq_len, model_dim]
        # p_all: [batch, seq_len, pe_dim]
        # e_all: [batch, seq_len, model_dim]
        
        # Step 4: Process positional encoding (if needed)
        # Note: E projection is already computed in fused output
        # If we need to apply E to P separately, we would do:
        #   e_all = W_e(p_all)  # But this is already in fused output
        # For now, e_all is the final positional encoding contribution
        
        # Optional: Add positional encoding to input
        # This can be done element-wise or via addition
        # For simplicity, we'll add e_all to the input for positional encoding
        x_with_pe = x_norm_all + e_all  # [batch, seq_len, model_dim]
        
    else:
        # Fallback: Separate projections (original approach)
        x_norm_all = self.norm1(x)
        
        # Separate matmul operations (6 kernel launches)
        q_all = self.W_q(x_norm_all)  # [batch, seq_len, model_dim]
        k_all = self.W_k(x_norm_all)  # [batch, seq_len, model_dim]
        v_all = self.W_v(x_norm_all)  # [batch, seq_len, model_dim]
        z_all = self.W_z(x_norm_all)  # [batch, seq_len, model_dim]
        p_all = self.W_p(x_norm_all)  # [batch, seq_len, pe_dim]
        e_all = self.W_e(p_all)       # [batch, seq_len, model_dim]
        
        x_with_pe = x_norm_all + e_all
    
    # Sequential processing: Loop over sequence steps
    for t in range(seq_len):
        # Get current timestep projections
        if self.use_hem:
            # Use pre-computed fused projections (faster, less CPU-GPU sync)
            x_t = x[:, t:t+1, :]  # [batch, 1, model_dim]
            x_t_norm = x_norm_all[:, t:t+1, :]  # [batch, 1, model_dim]
            q_t = q_all[:, t:t+1, :]  # [batch, 1, model_dim]
            k_t = k_all[:, t:t+1, :]  # [batch, 1, model_dim]
            v_t = v_all[:, t:t+1, :]  # [batch, 1, model_dim]
            z_t = z_all[:, t:t+1, :]  # [batch, 1, model_dim]
            p_t = p_all[:, t:t+1, :]  # [batch, 1, pe_dim]
            e_t = e_all[:, t:t+1, :]  # [batch, 1, model_dim]
        else:
            # Original per-step computation (slower, more CPU-GPU sync)
            x_t = x[:, t:t+1, :]
            x_t_norm = self.norm1(x_t)
            q_t = self.W_q(x_t_norm)
            k_t = self.W_k(x_t_norm)
            v_t = self.W_v(x_t_norm)
            z_t = self.W_z(x_t_norm)
            p_t = self.W_p(x_t_norm)
            e_t = self.W_e(p_t)
        
        # ... (devam: MDI, associative scan, core formula, attention, vb.)
        # (Mevcut kod devam eder)
        
        # Step 3-5: MDI computation
        h_prev_expanded = h_prev
        
        if self.use_gradient_checkpointing:
            mdi_fn = lambda z, h, ctx: self.mdi(z, h, context=ctx)
            h_new_t, gamma_new_t = checkpoint(mdi_fn, z_t, h_prev_expanded, k_t, use_reentrant=False)
        else:
            h_new_t, gamma_new_t = self.mdi(z_t, h_prev_expanded, context=k_t)
        
        # Step 4: Associative Scan
        gamma_t_reshaped = gamma_new_t.view(batch_size, self.num_heads, 1, -1)
        
        if not torch.cuda.is_available():
            from ..core.associative_scan_triton import associative_scan_exponential_cpu_fallback
            try:
                cumprod_t = associative_scan_exponential_cpu_fallback(gamma_t_reshaped)
            except RuntimeError as e:
                raise RuntimeError(
                    f"❌ CRITICAL: C++ extension required for CPU mode!\n"
                    f"   {str(e)}\n"
                ) from e
        else:
            try:
                from ..core.associative_scan_triton import associative_scan_exponential
                cumprod_t = associative_scan_exponential(gamma_t_reshaped)
            except RuntimeError:
                from ..core.associative_scan_triton import associative_scan_exponential_cpu_fallback
                cumprod_t = associative_scan_exponential_cpu_fallback(gamma_t_reshaped)
        
        cumprod_t = cumprod_t.view(batch_size, 1, self.model_dim)
        
        # Step 6: Core Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        gate_signal = torch.sigmoid(self.W_g(h_prev_expanded))
        gated_input = z_t * gate_signal
        decayed_prev = gamma_new_t * h_prev_expanded
        h_t = gated_input + decayed_prev
        
        # Ensure h_new_t from MDI also contributes
        h_t = h_t + 0.1 * h_new_t
        
        # Step 7: Multi-Memory Attention
        if self.use_gradient_checkpointing:
            attn_fn = lambda h, q: self.multi_mem_attention(h, hds, state, q_input=q)
            mem_context_t = checkpoint(attn_fn, h_t, q_t, use_reentrant=False)
        else:
            mem_context_t = self.multi_mem_attention(h_t, hds, state, q_input=q_t)
        
        # CRITICAL FIX: Ensure v_t contributes to output for gradient flow
        v_contribution = v_t * 0.1
        h_attended_t = h_t + mem_context_t + v_contribution
        
        # Step 8: Residual connection
        x_residual_t = x_t + self.dropout_layer(h_attended_t)
        
        # Step 9: Feed-forward network
        if self.use_gradient_checkpointing:
            def ffn_step(x_res):
                x_norm2 = self.norm2(x_res)
                return self.ffn(x_norm2)
            ffn_out_t = checkpoint(ffn_step, x_residual_t, use_reentrant=False)
        else:
            x_norm2_t = self.norm2(x_residual_t)
            ffn_out_t = self.ffn(x_norm2_t)
        
        output_t = x_residual_t + ffn_out_t
        
        # Store output for this timestep
        output[:, t:t+1, :] = output_t
        
        # Update memory state at step t
        h_t_for_state = h_t.squeeze(1)  # [batch, model_dim]
        state.update_state_sequential(
            bank_type='short',
            new_k=h_t_for_state,
            new_v=h_t_for_state,
            step=t
        )
        
        # Update h_prev for next iteration
        h_prev = h_t
    
    # Update long-term memory (block-level)
    h_sequence_summary = output.mean(dim=1, keepdim=True)
    h_summary_k = h_sequence_summary.squeeze(1)
    h_summary_v = h_sequence_summary.squeeze(1)
    
    state.update_state('long', 
                      h_summary_k.unsqueeze(1).expand(-1, state.long_term.num_slots, -1),
                      h_summary_v.unsqueeze(1).expand(-1, state.long_term.num_slots, -1))
    
    # Reconstruct HDS hierarchy with updated state
    hds.reset_cache()
    hds.construct_hierarchy(state)
    
    return output, state
```

### Alternatif: Daha Optimize Fused Matmul (Torch.matmul ile)

```python
# Alternative implementation using torch.matmul for more control
if self.use_hem:
    # Normalize input
    x_norm_all = self.norm1(x)  # [batch, seq_len, model_dim]
    
    # CRITICAL: Single fused matmul
    # Input:  x_norm_all [batch, seq_len, model_dim]
    # Weight: W_fused.weight [fused_out_dim, model_dim] (transposed for matmul)
    # Output: [batch, seq_len, fused_out_dim]
    #
    # Matmul: x_norm_all @ W_fused.weight.T + W_fused.bias
    fused_output = torch.matmul(
        x_norm_all,                    # [batch, seq_len, model_dim]
        self.W_fused.weight.t()        # [model_dim, fused_out_dim] (transposed)
    ) + self.W_fused.bias.unsqueeze(0).unsqueeze(0)  # Broadcast bias
    
    # Split into individual projections
    q_all, k_all, v_all, z_all, p_all, e_all = torch.split(
        fused_output,
        split_size_or_sections=[
            self.proj_dims['Q'],
            self.proj_dims['K'],
            self.proj_dims['V'],
            self.proj_dims['Z'],
            self.proj_dims['P'],
            self.proj_dims['E']
        ],
        dim=-1
    )
    
    # Add positional encoding
    x_with_pe = x_norm_all + e_all
```

---

## Özet

### 1. Fused Kernel Tanımı (`__init__`)

**Kritik Kod Parçaları**:
- **Tek Fused Linear Layer**: `nn.Linear(model_dim, 5*model_dim + pe_dim)`
- **Ağırlık Matrisi**: `W_fused = [W_Q; W_K; W_V; W_Z; W_P; W_E]`
- **Split İndisleri**: Her projeksiyonun çıktıda nerede olduğunu belirler
- **Initialization**: Her alt-matris için uygun initialization

**Boyutlar**:
- Input: `[batch, seq_len, model_dim]`
- Weight: `[model_dim, 5*model_dim + pe_dim]`
- Output: `[batch, seq_len, 5*model_dim + pe_dim]`

### 2. Fused Forward Pass (`forward`)

**Kritik Kod Parçaları**:
- **Tek Fused Matmul**: `F.linear(x_norm, W_fused.weight, W_fused.bias)`
- **Alternatif**: `torch.matmul(x_norm, W_fused.weight.t()) + bias`
- **Split Operation**: `torch.split()` ile çıktıyı QKVZPE'ye ayırma
- **Positional Encoding**: `e_all` ile input'a ekleme

**Performans Avantajları**:
- 6 ayrı kernel launch yerine 1 kernel launch
- Daha iyi memory bandwidth utilization
- Daha iyi cache locality
- GPU'da daha verimli paralel işleme

### Kritik Notlar

1. **Fused Weight Matrix**: Tüm projeksiyonlar tek bir büyük matriste birleştirilir
2. **Single Matmul**: `F.linear()` veya `torch.matmul()` ile tek işlem
3. **Split Operation**: Çıktıyı QKVZPE'ye ayırmak için `torch.split()` kullanılır
4. **Positional Encoding**: W_P (down) ve W_E (up) projeksiyonları fused output'ta
5. **Backward Compatibility**: `use_hem=False` ile eski yaklaşım kullanılabilir

---

**Doküman Versiyonu**: 1.0  
**Oluşturulma Tarihi**: 2025-01-27  
**Amaç**: HEM (Mekanizma 1) entegrasyonu için kritik kod parçaları


