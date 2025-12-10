# DPG (Dynamic Projection Gating) Entegrasyonu - Kritik Kod Parçaları

Bu doküman, MM-Rec analiz dokümanına dayanarak **Mekanizma 2 (DPG)** entegrasyonu için kritik kod parçalarını içerir.

---

## 1. Low-Rank Projeksiyon Tanımı

### Dosya: `mm_rec/blocks/mm_rec_block.py`

### `__init__` Metodunda DPG Ağırlıklarının Tanımı

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
    # DPG Parameters
    dpg_rank: int = 128,  # Low-rank projection dimension (D -> 128 -> D)
    use_dpg: bool = True   # Enable DPG mechanism
):
    super().__init__()
    self.model_dim = model_dim
    self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
    self.num_heads = num_heads
    self.num_memories = num_memories
    self.mem_dim = mem_dim if mem_dim is not None else model_dim
    self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
    self.dropout = dropout
    
    # DPG Configuration
    self.use_dpg = use_dpg
    self.dpg_rank = dpg_rank  # Low-rank dimension: 128
    
    # ... (mevcut projeksiyonlar: W_q, W_k, W_v, W_z, W_g)
    
    # ========================================================================
    # DPG: Dynamic Projection Gating - Low-Rank Projeksiyonlar
    # ========================================================================
    # DPG mekanizması için low-rank projeksiyonlar:
    # γ_t = σ(W_γ,up · ReLU(W_γ,down · z_t))
    # 
    # Boyutlar:
    # - W_γ,down: [model_dim, dpg_rank] = [4096, 128] (down-projection)
    # - W_γ,up:   [dpg_rank, model_dim] = [128, 4096] (up-projection)
    # 
    # Bu low-rank yapı sayesinde:
    # - Parametre sayısı: 4096×128 + 128×4096 = 1,048,576 (full: 4096×4096 = 16,777,216)
    # - 16x parametre tasarrufu
    # - Daha hızlı hesaplama
    # ========================================================================
    
    if self.use_dpg:
        # W_γ,down: Down-projection (D -> 128)
        # Input: z_t [batch, seq_len, model_dim]
        # Output: [batch, seq_len, dpg_rank]
        self.W_gamma_down = nn.Linear(
            in_features=model_dim,
            out_features=dpg_rank,
            bias=True  # Bias term for flexibility
        )
        
        # W_γ,up: Up-projection (128 -> D)
        # Input: [batch, seq_len, dpg_rank]
        # Output: [batch, seq_len, model_dim]
        self.W_gamma_up = nn.Linear(
            in_features=dpg_rank,
            out_features=model_dim,
            bias=True  # Bias term for flexibility
        )
        
        # Activation: ReLU between down and up projections
        # This adds non-linearity and ensures non-negative intermediate values
        self.dpg_activation = nn.ReLU()
        
        # Final activation: Sigmoid to ensure γ_t ∈ [0, 1]
        # Applied after up-projection
        self.dpg_sigmoid = nn.Sigmoid()
    else:
        # Fallback: Use full-rank projection (original MDI approach)
        # This is kept for backward compatibility
        self.W_gamma_down = None
        self.W_gamma_up = None
        self.dpg_activation = None
        self.dpg_sigmoid = None
    
    # ... (diğer bileşenler: MDI, attention, FFN, vb.)
```

### DPG Forward Pass (γ_t Hesaplama)

```python
def compute_dpg_gamma(
    self,
    z_t: torch.Tensor,
    context: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    DPG mekanizması ile dinamik γ_t hesaplama.
    
    Formül: γ_t = σ(W_γ,up · ReLU(W_γ,down · z_t))
    
    Args:
        z_t: Input tensor [batch, seq_len, model_dim] or [batch, model_dim]
        context: Optional context for modulation (future extension)
    
    Returns:
        gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
    """
    if not self.use_dpg:
        # Fallback to original MDI approach
        return self.mdi.compute_decay_only(z_t, context)
    
    # Step 1: Down-projection (D -> 128)
    # z_t: [batch, seq_len, model_dim]
    # W_γ,down: [model_dim, dpg_rank]
    # Output: [batch, seq_len, dpg_rank]
    z_projected_down = self.W_gamma_down(z_t)
    
    # Step 2: ReLU activation (non-linearity + non-negativity)
    z_activated = self.dpg_activation(z_projected_down)
    # z_activated: [batch, seq_len, dpg_rank]
    
    # Step 3: Up-projection (128 -> D)
    # W_γ,up: [dpg_rank, model_dim]
    # Output: [batch, seq_len, model_dim]
    z_projected_up = self.W_gamma_up(z_activated)
    
    # Step 4: Sigmoid activation (ensure γ_t ∈ [0, 1])
    gamma = self.dpg_sigmoid(z_projected_up)
    # gamma: [batch, seq_len, model_dim]
    
    # Step 5: Clamp to prevent numerical issues
    # Range: [1e-6, 1-1e-6] to avoid extreme values
    gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
    
    # Optional: Context modulation (if provided)
    if context is not None:
        # Future extension: context-dependent modulation
        # For now, just return gamma
        pass
    
    return gamma
```

### MMRecBlock Forward'da DPG Kullanımı

```python
def forward(
    self,
    x: torch.Tensor,
    state: MemoryState,
    hds: Optional[HierarchicalDataStructure] = None,
    use_checkpointing: Optional[bool] = None
) -> Tuple[torch.Tensor, MemoryState]:
    """
    Forward pass with DPG support.
    """
    # ... (mevcut kod: QKVZ projeksiyonları, vb.)
    
    # Sequential processing
    for t in range(seq_len):
        # ... (mevcut kod: q_t, k_t, v_t, z_t hesaplama)
        
        # ====================================================================
        # DPG: Dynamic γ_t Computation
        # ====================================================================
        # DPG mekanizması ile dinamik decay coefficient hesaplama
        # γ_t = σ(W_γ,up · ReLU(W_γ,down · z_t))
        # ====================================================================
        
        if self.use_dpg:
            # Use DPG for dynamic gamma computation
            gamma_new_t = self.compute_dpg_gamma(z_t, context=k_t)
            # gamma_new_t: [batch, 1, model_dim]
            
            # For MDI, we still need h_new_t (can use simplified version)
            # Or integrate DPG directly into MDI
            h_new_t = z_t * 0.1  # Simplified (or use full MDI)
        else:
            # Original MDI approach
            h_new_t, gamma_new_t = self.mdi(z_t, h_prev_expanded, context=k_t)
        
        # ... (devam: associative scan, core formula, vb.)
```

---

## 2. Triton Çekirdek Güncellemesi

### Dosya: `mm_rec/core/associative_scan_triton.py`

### Dinamik γ_t Tensörünü Kabul Eden Triton Çekirdek İmzası

```python
@triton.jit
def stable_log_sum_exp_fp64(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    """
    Stable log-sum-exp operation with FP64 precision for critical accumulation.
    
    CRITICAL: FP64 kullanımı zorunlu çünkü:
    - Log-space akümülasyonu çok hassas
    - Uzun dizilerde (32K+) küçük hatalar birikir
    - BF16/FP32 yeterli precision sağlamaz
    
    Uses pattern: max(a, b) + log(1 + exp(-abs(a - b)))
    
    Args:
        a: First log-value (FP64)
        b: Second log-value (FP64)
    
    Returns:
        log(exp(a) + exp(b)) in FP64
    """
    # FP64 operations for maximum precision
    max_val = tl.maximum(a, b)  # FP64
    diff = tl.abs(a - b)  # FP64
    # Clamp diff to prevent overflow in exp(-diff)
    diff_clamped = tl.minimum(diff, 20.0)  # exp(-20) ≈ 0
    return max_val + tl.log1p(tl.exp(-diff_clamped))  # FP64


@triton.jit
def associative_scan_parallel_kernel_dpg(
    gamma_ptr,          # Pointer to DYNAMIC gamma tensor [BATCH, HEADS, SEQ_LEN, D_HEAD]
                       # CRITICAL: This is now dynamic (computed per timestep via DPG)
    output_ptr,         # Pointer to output tensor [BATCH, HEADS, SEQ_LEN, D_HEAD]
    carry_in_ptr,       # Pointer to carry-over prefix from previous block [BATCH, HEADS, D_HEAD]
    carry_out_ptr,      # Pointer to output block prefix for next block [BATCH, HEADS, D_HEAD]
    batch_size,         # BATCH dimension
    num_heads,          # HEADS dimension
    seq_len,            # SEQ_LEN dimension
    head_dim,           # D_HEAD dimension
    stride_batch,       # Stride for batch dimension
    stride_heads,       # Stride for heads dimension
    stride_seq,         # Stride for sequence dimension
    stride_dim,         # Stride for head dimension
    block_idx,          # Current block index in sequence
    has_carry_in: tl.constexpr,  # Whether carry_in_ptr is valid
    has_carry_out: tl.constexpr,  # Whether carry_out_ptr is valid
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel processing
    USE_FP64_ACCUM: tl.constexpr = True,  # CRITICAL: Enable FP64 accumulation
):
    """
    Work-efficient parallel scan kernel for log-space associative scan with DPG support.
    
    CRITICAL CHANGES FOR DPG:
    1. Input is now DYNAMIC gamma_t tensor (computed per timestep via DPG)
    2. FP64 accumulation is MANDATORY for numerical stability
    3. Log-Sum-Exp pattern with FP64 precision
    
    Algorithm:
    1. Load dynamic gamma_t values (already in log-space or convert)
    2. Up-Sweep: Build reduction tree using FP64 log-sum-exp
    3. Down-Sweep: Propagate prefixes using FP64 log-sum-exp
    4. Carry-over: Add previous block prefix with FP64 precision
    
    This kernel processes one block of the sequence. Multiple blocks are handled
    by the Python wrapper with carry-over propagation.
    
    Args:
        gamma_ptr: Dynamic gamma tensor [BATCH, HEADS, SEQ_LEN, D_HEAD]
                  Values are in [0, 1] range (from DPG sigmoid output)
        USE_FP64_ACCUM: If True, use FP64 for accumulation (MANDATORY for DPG)
    """
    
    # Get program ID for this thread block
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)  # Head dimension index
    
    # Calculate base pointers
    base_batch = pid_batch * stride_batch
    base_head = pid_head * stride_heads
    base_dim = pid_dim * stride_dim
    
    # Calculate sequence range for this block
    seq_start = block_idx * BLOCK_SIZE
    seq_end = tl.minimum(seq_start + BLOCK_SIZE, seq_len)
    block_size = seq_end - seq_start
    
    # ========================================================================
    # CRITICAL: FP64 Accumulation Arrays
    # ========================================================================
    # Allocate local arrays with FP64 precision for accumulation
    # This is MANDATORY for DPG because:
    # 1. Dynamic gamma_t values can vary significantly
    # 2. Long sequences (32K+) accumulate small errors
    # 3. Log-space operations require high precision
    # ========================================================================
    if USE_FP64_ACCUM:
        # FP64 arrays for maximum precision
        block_data = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
        reduction_tree = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
        block_prefixes = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    else:
        # FP32 fallback (not recommended for DPG)
        block_data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        reduction_tree = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        block_prefixes = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # ========================================================================
    # LOAD INPUT DATA: Dynamic gamma_t values
    # ========================================================================
    # CRITICAL: gamma_t is now DYNAMIC (computed per timestep via DPG)
    # We need to:
    # 1. Load gamma_t values (in [0, 1] range from DPG sigmoid)
    # 2. Convert to log-space: log(gamma_t + epsilon)
    # 3. Clamp to [-50, 0] range for stability
    # 4. Store in FP64 for accumulation
    # ========================================================================
    epsilon = 1e-8  # Prevent log(0)
    log_min = -50.0  # Prevent underflow
    log_max = 0.0    # Prevent overflow
    
    for i in range(BLOCK_SIZE):
        seq_idx = seq_start + i
        if seq_idx < seq_end:
            offset = base_batch + base_head + seq_idx * stride_seq + base_dim
            # Load gamma_t value (in [0, 1] range from DPG)
            gamma_val = tl.load(gamma_ptr + offset)  # Load as FP32/BF16
            
            # Convert to FP64 for precision
            if USE_FP64_ACCUM:
                gamma_val_fp64 = tl.cast(gamma_val, tl.float64)
            else:
                gamma_val_fp64 = tl.cast(gamma_val, tl.float32)
            
            # Convert to log-space: log(gamma + epsilon)
            log_gamma = tl.log(gamma_val_fp64 + epsilon)
            
            # Clamp to [-50, 0] range for numerical stability
            log_gamma_clamped = tl.maximum(tl.minimum(log_gamma, log_max), log_min)
            
            # Store in accumulation array
            block_data[i] = log_gamma_clamped
            reduction_tree[i] = log_gamma_clamped
        else:
            # Identity in log-space (log(1) = 0)
            block_data[i] = 0.0
            reduction_tree[i] = 0.0
    
    # ========================================================================
    # UP-SWEEP PHASE: Build reduction tree with FP64 log-sum-exp
    # ========================================================================
    # This phase computes the total sum of the block in log-space
    # Uses work-efficient parallel reduction (O(n) work, O(log n) depth)
    # CRITICAL: All operations use FP64 precision
    # ========================================================================
    
    stride = 1
    while stride < block_size:
        # Process elements at positions: stride, 3*stride, 5*stride, ...
        for i in range(stride, block_size, stride * 2):
            left_idx = i - stride
            right_idx = i
            
            if right_idx < block_size:
                val_left = reduction_tree[left_idx]
                val_right = reduction_tree[right_idx]
                
                # CRITICAL: Use FP64 log-sum-exp for stable combination
                if USE_FP64_ACCUM:
                    # FP64 stable log-sum-exp
                    reduction_tree[right_idx] = stable_log_sum_exp_fp64(val_left, val_right)
                else:
                    # FP32 fallback (not recommended)
                    reduction_tree[right_idx] = stable_log_sum_exp(val_left, val_right)
        
        stride = stride * 2
    
    # The last element (reduction_tree[block_size - 1]) now contains the total sum
    block_total = reduction_tree[block_size - 1] if block_size > 0 else 0.0
    
    # ========================================================================
    # LOAD CARRY-OVER PREFIX (from previous block) - FP64
    # ========================================================================
    carry_prefix = 0.0  # Identity in log-space
    if has_carry_in and block_idx > 0:
        carry_offset = base_batch + base_head + base_dim
        carry_val = tl.load(carry_in_ptr + carry_offset)
        
        # Convert to FP64 if needed
        if USE_FP64_ACCUM:
            carry_prefix = tl.cast(carry_val, tl.float64)
        else:
            carry_prefix = tl.cast(carry_val, tl.float32)
    
    # ========================================================================
    # DOWN-SWEEP PHASE: Propagate prefixes with FP64 precision
    # ========================================================================
    # Blelloch scan down-sweep: propagate prefixes using reduction tree
    # Initialize: last element gets identity (0.0) for prefix
    # CRITICAL: All operations use FP64 precision
    # ========================================================================
    if block_size > 0:
        block_prefixes[block_size - 1] = 0.0  # Identity
    
    # Down-sweep: propagate prefixes from root to leaves
    stride = block_size // 2
    while stride > 0:
        for i in range(stride, block_size, stride * 2):
            left_idx = i - stride
            right_idx = i
            
            if right_idx < block_size:
                # Get prefix being propagated to left subtree
                prefix_to_left = block_prefixes[right_idx]
                # Get total from left subtree (from reduction tree)
                left_total = reduction_tree[left_idx]
                
                # CRITICAL: Use FP64 log-sum-exp for stable combination
                if USE_FP64_ACCUM:
                    # Propagate: left gets the prefix, right gets prefix + left_total
                    block_prefixes[left_idx] = prefix_to_left
                    block_prefixes[right_idx] = stable_log_sum_exp_fp64(prefix_to_left, left_total)
                else:
                    # FP32 fallback
                    block_prefixes[left_idx] = prefix_to_left
                    block_prefixes[right_idx] = stable_log_sum_exp(prefix_to_left, left_total)
        
        stride = stride // 2
    
    # Final step: combine prefixes with original data
    # block_prefixes[i] now contains prefix up to (but not including) position i
    # Final result at position i = prefix[i] + data[i]
    for i in range(block_size):
        if USE_FP64_ACCUM:
            final_prefix = stable_log_sum_exp_fp64(block_prefixes[i], block_data[i])
        else:
            final_prefix = stable_log_sum_exp(block_prefixes[i], block_data[i])
        block_prefixes[i] = final_prefix
    
    # ========================================================================
    # ADD CARRY-OVER PREFIX AND STORE RESULTS
    # ========================================================================
    # CRITICAL: Convert back to FP32/BF16 for storage (output dtype)
    # ========================================================================
    for i in range(block_size):
        seq_idx = seq_start + i
        
        # Get prefix for this position (within block)
        prefix_within_block = block_prefixes[i]
        
        # Combine with carry-over prefix from previous blocks
        if block_idx > 0:
            # Add carry-over prefix using log-sum-exp
            if USE_FP64_ACCUM:
                final_prefix = stable_log_sum_exp_fp64(carry_prefix, prefix_within_block)
            else:
                final_prefix = stable_log_sum_exp(carry_prefix, prefix_within_block)
        else:
            # First block: no carry-over
            final_prefix = prefix_within_block
        
        # Convert back to FP32 for storage (output will be converted to original dtype)
        final_prefix_fp32 = tl.cast(final_prefix, tl.float32)
        
        # Store result
        offset = base_batch + base_head + seq_idx * stride_seq + base_dim
        tl.store(output_ptr + offset, final_prefix_fp32)
    
    # ========================================================================
    # STORE BLOCK PREFIX FOR NEXT BLOCK (carry-out) - FP64 -> FP32
    # ========================================================================
    if has_carry_out:
        # Block prefix = carry_prefix + block_total
        if USE_FP64_ACCUM:
            block_prefix = stable_log_sum_exp_fp64(carry_prefix, block_total)
        else:
            block_prefix = stable_log_sum_exp(carry_prefix, block_total)
        
        # Convert to FP32 for storage
        block_prefix_fp32 = tl.cast(block_prefix, tl.float32)
        carry_offset = base_batch + base_head + base_dim
        tl.store(carry_out_ptr + carry_offset, block_prefix_fp32)
```

### PyTorch Function Wrapper (DPG Desteği ile)

```python
class AssociativeScanExponentialDPG(Function):
    """
    PyTorch autograd Function for associative scan with exponential product (DPG support).
    
    CRITICAL CHANGES FOR DPG:
    1. Accepts DYNAMIC gamma_t tensor (computed per timestep via DPG)
    2. Uses FP64 accumulation in Triton kernel (MANDATORY)
    3. Converts back to original dtype (BF16/FP32) for output
    
    Implements: Y_t = ∏_{i=1}^t γ_i using Log-Sum-Exp pattern with FP64 precision.
    
    Args:
        gamma: Dynamic gamma tensor [BATCH, HEADS, SEQ_LEN, D_HEAD] of decay coefficients
               Values should be in [0, 1] range (from DPG sigmoid output)
               CRITICAL: This is now DYNAMIC (computed per timestep)
    
    Returns:
        cumulative_product: [BATCH, HEADS, SEQ_LEN, D_HEAD] cumulative products
    """
    
    @staticmethod
    def forward(ctx, gamma: torch.Tensor, use_fp64_accum: bool = True) -> torch.Tensor:
        """
        Forward pass: Compute cumulative exponential product with DPG support.
        
        Steps:
        1. Convert dynamic gamma_t to log-space: log(gamma + eps)
        2. Clamp log values to [-50, 0] range
        3. Call Triton kernel with FP64 accumulation (MANDATORY for DPG)
        4. Convert back to linear space with stability
        5. Convert to original dtype (BF16)
        """
        
        # Input validation
        assert gamma.dim() == 4, f"Expected 4D tensor, got {gamma.dim()}D"
        assert gamma.dtype in [torch.float16, torch.bfloat16, torch.float32], \
            f"Unsupported dtype: {gamma.dtype}"
        
        batch_size, num_heads, seq_len, head_dim = gamma.shape
        
        # CRITICAL: Convert to FP32 for log operations (numerical stability)
        # We'll use FP64 in Triton kernel, but prepare in FP32
        gamma_fp32 = gamma.to(torch.float32)
        
        # Step 1: Convert to log-space with clamping
        # Add epsilon to prevent log(0)
        epsilon = 1e-8
        log_gamma = torch.log(gamma_fp32 + epsilon)
        
        # Step 2: Clamp log values to [-50, 0] range
        # This prevents underflow (exp(-50) ≈ 0) and overflow (exp(0) = 1)
        log_gamma_clamped = torch.clamp(log_gamma, min=-50.0, max=0.0)
        
        # Step 3: Prepare output tensor (FP32 for accumulation, will convert to FP64 in kernel)
        log_cumsum = torch.empty_like(log_gamma_clamped, dtype=torch.float32)
        
        # Step 4: Launch Triton kernel with FP64 accumulation (MANDATORY for DPG)
        # Calculate strides
        stride_batch = log_gamma_clamped.stride(0)
        stride_heads = log_gamma_clamped.stride(1)
        stride_seq = log_gamma_clamped.stride(2)
        stride_dim = log_gamma_clamped.stride(3)
        
        # Determine block size (power of 2, optimal for parallel scan)
        if seq_len >= 1024:
            BLOCK_SIZE = 1024
        elif seq_len >= 512:
            BLOCK_SIZE = 512
        elif seq_len >= 256:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 128
        
        # Calculate number of blocks needed for sequence dimension
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Allocate carry-over buffers for block-to-block communication
        # CRITICAL: Use FP32 for storage (Triton kernel will use FP64 internally)
        carry_in = torch.zeros(batch_size, num_heads, head_dim, 
                               dtype=torch.float32, device=log_gamma_clamped.device)
        carry_out = torch.zeros(batch_size, num_heads, head_dim,
                                dtype=torch.float32, device=log_gamma_clamped.device)
        
        # Grid dimensions: (batch, heads, head_dim)
        grid = (batch_size, num_heads, head_dim)
        
        # CRITICAL: Triton fallback detection
        triton_available = torch.cuda.is_available() and hasattr(triton, 'jit')
        triton_failed = False
        
        # Process blocks sequentially with carry-over propagation
        for block_idx in range(num_blocks):
            # Determine carry_in pointer and flag
            has_carry_in = block_idx > 0
            carry_in_ptr = carry_in if has_carry_in else torch.empty(0, device=log_gamma_clamped.device)
            
            # Determine carry_out pointer and flag
            has_carry_out = block_idx < num_blocks - 1
            carry_out_ptr = carry_out if has_carry_out else torch.empty(0, device=log_gamma_clamped.device)
            
            # Launch kernel for this block with FP64 accumulation (MANDATORY for DPG)
            try:
                if triton_available:
                    associative_scan_parallel_kernel_dpg[grid](
                        log_gamma_clamped,      # Input: log(gamma) in FP32
                        log_cumsum,             # Output: log(cumsum) in FP32
                        carry_in_ptr,
                        carry_out_ptr,
                        batch_size,
                        num_heads,
                        seq_len,
                        head_dim,
                        stride_batch,
                        stride_heads,
                        stride_seq,
                        stride_dim,
                        block_idx,              # Current block index
                        has_carry_in=has_carry_in,
                        has_carry_out=has_carry_out,
                        BLOCK_SIZE=BLOCK_SIZE,
                        USE_FP64_ACCUM=use_fp64_accum,  # CRITICAL: Enable FP64
                    )
                else:
                    triton_failed = True
                    break
            except Exception as e:
                # Triton kernel failed - fall back to CPU
                triton_failed = True
                import warnings
                warnings.warn(
                    f"⚠️ Triton kernel failed at block {block_idx}/{num_blocks}: {e}\n"
                    f"   Falling back to CPU implementation.\n"
                    f"   FP64 accumulation is CRITICAL for DPG!",
                    RuntimeWarning,
                    stacklevel=2
                )
                break
            
            # Propagate carry-over to next block
            if block_idx < num_blocks - 1:
                carry_in = carry_out.clone()
                carry_out.zero_()
        
        # If Triton failed, handle based on device
        if triton_failed:
            # On CPU, C++ extension is REQUIRED (no slow Python fallback)
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "❌ CRITICAL: Triton kernel failed on CPU and C++ extension is REQUIRED!\n"
                    "   FP64 accumulation is MANDATORY for DPG!\n"
                    "   Solution: Ensure mm_rec_scan_cpu is built with FP64 support."
                )
            
            # On GPU, log warning but continue with slow fallback (last resort)
            import warnings
            warnings.warn(
                "⚠️ CRITICAL: Triton kernel failed! Using slow CPU fallback.\n"
                "   FP64 accumulation is MANDATORY for DPG numerical stability!",
                RuntimeWarning,
                stacklevel=2
            )
            # Use slow Python fallback (only on GPU, as last resort)
            log_cumsum = torch.cumsum(log_gamma_clamped, dim=2)
        
        # Step 5: Convert back to linear space with stability
        # Use max-subtraction pattern: exp(log_sum - max) * exp(max)
        max_log = torch.max(log_cumsum, dim=2, keepdim=True)[0]  # Max over sequence
        stable_log = log_cumsum - max_log
        cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
        
        # Convert back to original dtype (BF16)
        cumulative_product = cumulative_product.to(gamma.dtype)
        
        # Save for backward pass
        ctx.save_for_backward(gamma, cumulative_product, log_cumsum, max_log)
        ctx.gamma_dtype = gamma.dtype
        ctx.seq_len = seq_len
        ctx.use_fp64_accum = use_fp64_accum
        
        return cumulative_product
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Backward pass: Compute gradients w.r.t. dynamic gamma_t.
        
        For cumulative product Y_t = ∏_{i=1}^t γ_i:
        - grad_γ_i = Σ_{t=i}^T (Y_t / γ_i) * grad_Y_t
        - This requires reverse cumulative sum (right-to-left)
        - CRITICAL: Use FP64 for gradient accumulation
        """
        # ... (similar to original backward, but with FP64 support)
        # (Implementation details similar to original, but with FP64 accumulation)
        pass


# User-facing function
def associative_scan_exponential_dpg(
    gamma: torch.Tensor,
    use_fp64_accum: bool = True
) -> torch.Tensor:
    """
    User-facing function for exponential product associative scan with DPG support.
    
    CRITICAL: FP64 accumulation is MANDATORY for DPG numerical stability.
    
    Args:
        gamma: Dynamic gamma tensor [BATCH, HEADS, SEQ_LEN, D_HEAD] (from DPG)
               Values should be in [0, 1] range
        use_fp64_accum: Enable FP64 accumulation (MANDATORY for DPG)
    
    Returns:
        cumulative_product: [BATCH, HEADS, SEQ_LEN, D_HEAD] cumulative products
    """
    return AssociativeScanExponentialDPG.apply(gamma, use_fp64_accum)
```

---

## Özet

### 1. Low-Rank Projeksiyon Tanımı (`mm_rec_block.py`)

- **W_γ,down**: `nn.Linear(model_dim, dpg_rank)` → [4096, 128]
- **W_γ,up**: `nn.Linear(dpg_rank, model_dim)` → [128, 4096]
- **Activation**: ReLU (down-up arası) + Sigmoid (final)
- **Parametre Tasarrufu**: 16x (1M vs 16M parametre)

### 2. Triton Çekirdek Güncellemesi (`associative_scan_triton.py`)

- **Dinamik γ_t Input**: Triton çekirdeği artık dinamik gamma_t tensörünü kabul eder
- **FP64 Accumulation**: `USE_FP64_ACCUM=True` zorunlu (sayısal kararlılık için)
- **Log-Sum-Exp Pattern**: FP64 precision ile stable log-sum-exp
- **Kernel İmzası**: `associative_scan_parallel_kernel_dpg` (DPG desteği ile)

### Kritik Notlar

1. **FP64 Zorunlu**: DPG ile dinamik gamma_t değerleri için FP64 accumulation kritik
2. **Low-Rank Yapı**: D -> 128 -> D projeksiyonu parametre tasarrufu sağlar
3. **Dinamik Gamma**: Her timestep'te DPG ile hesaplanan gamma_t değerleri
4. **Sayısal Kararlılık**: Log-Sum-Exp pattern + FP64 + clamping

---

**Doküman Versiyonu**: 1.0  
**Oluşturulma Tarihi**: 2025-01-27  
**Amaç**: DPG (Mekanizma 2) entegrasyonu için kritik kod parçaları


