# UBÖO (Mekanizma 3) Entegrasyonu - Kritik Kod Parçaları

Bu doküman, MM-Rec analiz dokümanına dayanarak **Mekanizma 3 (UBÖO - Unbiased Backpropagation with Error Orthogonalization)** entegrasyonu için kritik kod parçalarını içerir.

---

## 1. Gradyan İzolasyonu (Gradient Isolation)

### Dosya: `mm_rec/core/mdi.py`

### Planlama Hatası Hesaplama ve `detach()` Kullanımı

```python
class MemoryDecayIntegration(nn.Module):
    """
    Memory Decay/Integration mechanism for MM-Rec with UBÖO support.
    
    Implements the core formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    
    UBÖO Extension:
    - Computes Planning Error (P_error) for auxiliary loss
    - Uses gradient isolation via detach() to prevent gradient flow through M_updated
    - Enables unbiased backpropagation through error orthogonalization
    """
    
    def __init__(
        self,
        model_dim: int,
        inner_dim: Optional[int] = None,
        use_context_modulation: bool = True,
        # UBÖO Parameters
        use_uboo: bool = True,           # Enable UBÖO mechanism
        planning_error_dim: Optional[int] = None  # Dimension for planning error
    ):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim if inner_dim is not None else model_dim // 4
        self.use_context_modulation = use_context_modulation
        
        # UBÖO Configuration
        self.use_uboo = use_uboo
        self.planning_error_dim = planning_error_dim if planning_error_dim is not None else model_dim
        
        # Gating weight: Controls how much of new (z_t) vs old (h_{t-1}) to combine
        self.W_g = nn.Linear(model_dim * 2, model_dim)
        
        # Decay weight: Learns decay coefficient γ
        self.W_gamma = nn.Sequential(
            nn.Linear(model_dim, self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, model_dim),
            nn.Sigmoid()
        )
        
        # Context modulation (optional)
        if self.use_context_modulation:
            self.W_context = nn.Sequential(
                nn.Linear(model_dim, self.inner_dim),
                nn.GELU(),
                nn.Linear(self.inner_dim, model_dim),
                nn.Sigmoid()
            )
        else:
            self.W_context = None
        
        # ========================================================================
        # UBÖO: Planning Error Computation Components
        # ========================================================================
        # Planning Error (P_error) measures the discrepancy between:
        # - M_updated: Updated memory state (with gradients)
        # - M_target: Target memory state (computed from planning)
        # 
        # CRITICAL: M_updated must be detached to prevent gradient flow through
        # the planning error computation, enabling unbiased backpropagation.
        # ========================================================================
        
        if self.use_uboo:
            # Planning error projection: Maps memory state to error space
            # Input: M_updated (detached) [batch, seq_len, model_dim]
            # Output: P_error [batch, seq_len, planning_error_dim]
            self.W_planning_error = nn.Sequential(
                nn.Linear(model_dim, self.planning_error_dim),
                nn.GELU(),
                nn.Linear(self.planning_error_dim, self.planning_error_dim)
            )
            
            # Target memory projection: Computes target from planning
            # This is used to compute the planning error target
            self.W_planning_target = nn.Sequential(
                nn.Linear(model_dim, self.planning_error_dim),
                nn.GELU(),
                nn.Linear(self.planning_error_dim, self.planning_error_dim)
            )
        else:
            self.W_planning_error = None
            self.W_planning_target = None
    
    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_auxiliary_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: Compute new state h_t, decay coefficient γ, and auxiliary loss.
        
        Implements: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        
        UBÖO Extension:
        - Computes Planning Error (P_error) for auxiliary loss
        - Uses detach() to isolate gradients
        
        Args:
            z_t: New input [batch, seq_len, model_dim] or [batch, model_dim]
            h_prev: Previous hidden state [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context tensor for modulation
            return_auxiliary_loss: If True, return auxiliary loss (L_Aux)
        
        Returns:
            Tuple of (h_new, gamma, L_Aux):
                - h_new: New hidden state [batch, seq_len, model_dim] or [batch, model_dim]
                - gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
                - L_Aux: Auxiliary loss (Planning Error) [batch, seq_len] or scalar (if return_auxiliary_loss=True)
        """
        # Ensure same shape
        assert z_t.shape == h_prev.shape, f"Shape mismatch: z_t {z_t.shape} vs h_prev {h_prev.shape}"
        
        # Compute gating signal: g = σ(W_g · [z_t, h_prev])
        concat_input = torch.cat([z_t, h_prev], dim=-1)
        gate = torch.sigmoid(self.W_g(concat_input))
        
        # Gated integration: h_tilde = (1 - g) ⊙ h_prev + g ⊙ z_t
        h_tilde = (1 - gate) * h_prev + gate * z_t
        
        # Compute decay coefficient: γ = σ(W_γ · z_t)
        gamma = self.W_gamma(z_t)
        
        # Apply context modulation if enabled
        if self.use_context_modulation and context is not None:
            modulation = self.W_context(context)
            gamma = gamma * modulation
        
        # Clamp decay coefficient to prevent numerical issues
        gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
        
        # Apply decay: h_new = h_tilde + γ ⊙ h_prev (residual-like)
        h_new = h_tilde + gamma * h_prev
        
        # ========================================================================
        # UBÖO: Planning Error Computation with Gradient Isolation
        # ========================================================================
        # CRITICAL: M_updated (h_new) must be detached to prevent gradient flow
        # through the planning error computation. This enables unbiased backpropagation.
        # 
        # Planning Error Formula:
        #   P_error = ||W_planning_error(M_updated.detach()) - W_planning_target(M_target)||²
        # 
        # Where:
        #   - M_updated: Updated memory state (h_new) - DETACHED
        #   - M_target: Target memory state (computed from planning)
        # 
        # The detach() operation ensures that:
        #   1. Gradients from P_error do not flow back through M_updated
        #   2. Planning error is computed independently of the main forward pass
        #   3. Unbiased backpropagation is maintained
        # ========================================================================
        
        L_Aux = None
        if self.use_uboo and return_auxiliary_loss:
            # Step 1: CRITICAL - Detach M_updated to isolate gradients
            # M_updated = h_new (updated memory state)
            # We detach it to prevent gradient flow through planning error computation
            M_updated_detached = h_new.detach()  # [batch, seq_len, model_dim] or [batch, model_dim]
            # CRITICAL: .detach() creates a new tensor that shares storage but
            # does not require gradients. This breaks the computational graph
            # for the planning error computation.
            
            # Step 2: Compute planning error projection from detached M_updated
            # This projection operates on the detached tensor, so no gradients
            # flow back through M_updated
            P_error_pred = self.W_planning_error(M_updated_detached)
            # P_error_pred: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 3: Compute target from planning (using original h_new with gradients)
            # The target is computed from the planning mechanism, which may use
            # the original h_new (with gradients) or a separate planning computation
            # For simplicity, we use h_prev as the planning target (can be customized)
            M_target = h_prev  # [batch, seq_len, model_dim] or [batch, model_dim]
            P_error_target = self.W_planning_target(M_target)
            # P_error_target: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 4: Compute Planning Error (P_error)
            # P_error = ||P_error_pred - P_error_target||²
            # This is the auxiliary loss that measures planning discrepancy
            P_error = P_error_pred - P_error_target
            # P_error: [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
            
            # Step 5: Compute auxiliary loss (L_Aux)
            # L_Aux = mean(||P_error||²) over sequence and error dimensions
            # This is a scalar loss that will be added to the main loss
            if P_error.dim() == 3:
                # [batch, seq_len, planning_error_dim]
                L_Aux = torch.mean(P_error ** 2, dim=-1)  # [batch, seq_len]
                L_Aux = torch.mean(L_Aux)  # Scalar
            else:
                # [batch, planning_error_dim]
                L_Aux = torch.mean(P_error ** 2, dim=-1)  # [batch]
                L_Aux = torch.mean(L_Aux)  # Scalar
            
            # CRITICAL: L_Aux is computed from detached M_updated, so:
            # - Gradients from L_Aux do NOT flow back through M_updated
            # - Gradients from L_Aux DO flow back through W_planning_error and W_planning_target
            # - This enables unbiased backpropagation through error orthogonalization
        
        return h_new, gamma, L_Aux
    
    def compute_planning_error(
        self,
        M_updated: torch.Tensor,
        M_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Planning Error (P_error) with gradient isolation.
        
        CRITICAL: M_updated is detached to prevent gradient flow through
        the planning error computation.
        
        Args:
            M_updated: Updated memory state [batch, seq_len, model_dim] or [batch, model_dim]
            M_target: Target memory state [batch, seq_len, model_dim] or [batch, model_dim]
        
        Returns:
            P_error: Planning error [batch, seq_len, planning_error_dim] or [batch, planning_error_dim]
        """
        if not self.use_uboo:
            return None
        
        # CRITICAL: Detach M_updated to isolate gradients
        # This prevents gradient flow through the planning error computation
        M_updated_detached = M_updated.detach()
        
        # Compute planning error projections
        P_error_pred = self.W_planning_error(M_updated_detached)
        P_error_target = self.W_planning_target(M_target)
        
        # Compute Planning Error
        P_error = P_error_pred - P_error_target
        
        return P_error
    
    def compute_decay_only(
        self,
        z_t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute only decay coefficient γ without state update.
        
        Useful for Associative Scan where we need γ values separately.
        
        Args:
            z_t: New input [batch, seq_len, model_dim] or [batch, model_dim]
            context: Optional context tensor
        
        Returns:
            gamma: Decay coefficient [batch, seq_len, model_dim] or [batch, model_dim]
        """
        gamma = self.W_gamma(z_t)
        
        if self.use_context_modulation and context is not None:
            modulation = self.W_context(context)
            gamma = gamma * modulation
        
        gamma = torch.clamp(gamma, min=1e-6, max=1.0 - 1e-6)
        return gamma
```

---

## 2. Auxiliary Loss Toplama

### Dosya: `mm_rec/model.py`

### Tüm Katmanlardan Gelen L_Aux Hatalarının Toplanması ve Lambda_P ile Ölçeklenmesi

```python
class MMRecModel(nn.Module):
    """
    Complete MM-Rec model architecture with UBÖO support.
    
    UBÖO Extension:
    - Collects auxiliary losses (L_Aux) from all layers
    - Scales auxiliary losses with lambda_P factor
    - Adds scaled auxiliary losses to main loss
    """
    
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 4096,
        num_layers: int = 24,
        num_heads: int = 8,
        num_memories: int = 1,
        mem_dim: Optional[int] = None,
        max_seq_len: int = 32768,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        # UBÖO Parameters
        use_uboo: bool = True,           # Enable UBÖO mechanism
        lambda_P: float = 0.1           # Scaling factor for auxiliary loss
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memories = num_memories
        self.mem_dim = mem_dim if mem_dim is not None else model_dim
        self.max_seq_len = max_seq_len
        self.ffn_dim = ffn_dim if ffn_dim is not None else model_dim * 4
        self.dropout = dropout
        
        # UBÖO Configuration
        self.use_uboo = use_uboo
        self.lambda_P = lambda_P  # Scaling factor for auxiliary loss
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, model_dim)
        
        # Initialize memory state
        M = 1024  # Long-term memory size
        memory_dtype = torch.float32
        
        short_term_config = {
            'k_dim': model_dim,
            'v_dim': model_dim,
            'num_slots': max_seq_len,
            'dtype': memory_dtype
        }
        
        long_term_config = {
            'k_dim': self.mem_dim,
            'v_dim': self.mem_dim,
            'num_slots': M,
            'dtype': memory_dtype
        }
        
        self.memory_config = {
            'short_term': short_term_config,
            'long_term': long_term_config
        }
        
        # MM-Rec blocks (24 layers as per spec)
        # CRITICAL: Each block's MDI module has UBÖO support
        self.blocks = nn.ModuleList([
            MMRecBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_memories=num_memories,
                mem_dim=self.mem_dim,
                ffn_dim=self.ffn_dim,
                dropout=dropout,
                use_uboo=use_uboo  # Pass UBÖO flag to blocks
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(model_dim)
        
        # Output head (language modeling)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Tie weights: output embedding = input embedding
        self.embedding.weight = self.lm_head.weight
        
        # Performance optimization flags
        self.use_gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None,
        chunk_size: Optional[int] = None,
        return_auxiliary_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through complete MM-Rec model with UBÖO support.
        
        CRITICAL: UBÖO mekanizması ile auxiliary loss (L_Aux) toplama ve ölçekleme.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_states: Optional list of MemoryState instances (one per layer)
            chunk_size: Chunk size for long sequences
            return_auxiliary_loss: If True, return auxiliary loss (L_Aux_total)
        
        Returns:
            Tuple of (logits, L_Aux_total):
                - logits: Language modeling logits [batch, seq_len, vocab_size]
                - L_Aux_total: Total auxiliary loss (scalar) if return_auxiliary_loss=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Determine chunk size
        if chunk_size is None:
            if seq_len > 32768:
                chunk_size = 8192
            else:
                chunk_size = None
        
        # Create memory states if not provided
        if memory_states is None:
            memory_states = [
                self.create_memory_state(batch_size, device)
                for _ in range(self.num_layers)
            ]
        
        # ========================================================================
        # UBÖO: Auxiliary Loss Collection
        # ========================================================================
        # CRITICAL: Collect auxiliary losses (L_Aux) from all layers
        # Each layer's MDI module computes a planning error (P_error) which
        # contributes to the auxiliary loss. These losses are collected and
        # scaled with lambda_P factor before being added to the main loss.
        # ========================================================================
        
        # List to collect auxiliary losses from all layers
        auxiliary_losses = []  # Will contain L_Aux from each layer
        
        # CHUNKING: Process long sequences in chunks
        if chunk_size is not None and seq_len > chunk_size:
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            all_logits = []
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                
                # Embedding
                x_chunk = self.embedding(chunk_input_ids)
                
                # Forward through all MM-Rec blocks
                new_memory_states = []
                for i, block in enumerate(self.blocks):
                    use_checkpointing = getattr(self, 'use_gradient_checkpointing', False)
                    if use_checkpointing and i >= len(self.blocks) // 2:
                        from torch.utils.checkpoint import checkpoint
                        def block_forward(x_in, state_in):
                            return block(x_in, state_in, use_checkpointing=False, return_auxiliary_loss=self.use_uboo)
                        result = checkpoint(
                            block_forward, x_chunk, memory_states[i], use_reentrant=False
                        )
                        if self.use_uboo and return_auxiliary_loss:
                            x_chunk, updated_state, L_Aux_layer = result
                            # Collect auxiliary loss from this layer
                            if L_Aux_layer is not None:
                                auxiliary_losses.append(L_Aux_layer)
                        else:
                            x_chunk, updated_state = result
                    else:
                        result = block(x_chunk, memory_states[i], return_auxiliary_loss=self.use_uboo)
                        if self.use_uboo and return_auxiliary_loss:
                            x_chunk, updated_state, L_Aux_layer = result
                            # Collect auxiliary loss from this layer
                            if L_Aux_layer is not None:
                                auxiliary_losses.append(L_Aux_layer)
                        else:
                            x_chunk, updated_state = result
                    
                    memory_states[i] = updated_state
                    new_memory_states.append(updated_state)
                
                # Final normalization
                x_chunk = self.norm(x_chunk)
                
                # Output head
                logits_chunk = self.lm_head(x_chunk)
                all_logits.append(logits_chunk)
            
            # Concatenate all chunk logits
            logits = torch.cat(all_logits, dim=1)
            
        else:
            # NO CHUNKING: Process entire sequence at once
            x = self.embedding(input_ids)
            
            # Forward through all MM-Rec blocks
            new_memory_states = []
            for i, block in enumerate(self.blocks):
                use_checkpointing = getattr(self, 'use_gradient_checkpointing', False)
                if use_checkpointing and i >= len(self.blocks) // 2:
                    from torch.utils.checkpoint import checkpoint
                    def block_forward(x_in, state_in):
                        return block(x_in, state_in, use_checkpointing=False, return_auxiliary_loss=self.use_uboo)
                    result = checkpoint(block_forward, x, memory_states[i], use_reentrant=False)
                    if self.use_uboo and return_auxiliary_loss:
                        x, updated_state, L_Aux_layer = result
                        # Collect auxiliary loss from this layer
                        if L_Aux_layer is not None:
                            auxiliary_losses.append(L_Aux_layer)
                    else:
                        x, updated_state = result
                else:
                    result = block(x, memory_states[i], return_auxiliary_loss=self.use_uboo)
                    if self.use_uboo and return_auxiliary_loss:
                        x, updated_state, L_Aux_layer = result
                        # Collect auxiliary loss from this layer
                        if L_Aux_layer is not None:
                            auxiliary_losses.append(L_Aux_layer)
                    else:
                        x, updated_state = result
                new_memory_states.append(updated_state)
            
            # Final normalization
            x = self.norm(x)
            
            # Output head
            logits = self.lm_head(x)
        
        # ========================================================================
        # UBÖO: Auxiliary Loss Aggregation and Scaling
        # ========================================================================
        # CRITICAL: Sum all auxiliary losses from all layers and scale with lambda_P
        # 
        # Formula:
        #   L_Aux_total = λ_P * Σ_{l=1}^{L} L_Aux^{(l)}
        # 
        # Where:
        #   - L_Aux^{(l)}: Auxiliary loss from layer l
        #   - λ_P: Scaling factor (lambda_P)
        #   - L: Number of layers (num_layers)
        # 
        # The scaled auxiliary loss is then added to the main loss:
        #   L_total = L_main + L_Aux_total
        # 
        # This enables unbiased backpropagation through error orthogonalization.
        # ========================================================================
        
        L_Aux_total = None
        if self.use_uboo and return_auxiliary_loss:
            if len(auxiliary_losses) > 0:
                # Sum all auxiliary losses from all layers
                # Each L_Aux_layer is a scalar (mean over batch and sequence)
                L_Aux_sum = sum(auxiliary_losses)  # Sum of scalars
                
                # CRITICAL: Scale with lambda_P factor
                # This controls the contribution of auxiliary loss to the total loss
                # Typical values: lambda_P ∈ [0.01, 0.1]
                L_Aux_total = self.lambda_P * L_Aux_sum
                
                # L_Aux_total is now a scalar that can be added to the main loss
            else:
                # No auxiliary losses collected (should not happen if UBÖO is enabled)
                L_Aux_total = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        if return_auxiliary_loss:
            return logits, L_Aux_total
        else:
            return logits
    
    def create_memory_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """
        Create a new MemoryState for a batch.
        
        Args:
            batch_size: Batch size
            device: Device to create state on
        
        Returns:
            MemoryState instance
        """
        return MemoryState(
            short_term_config=self.memory_config['short_term'],
            long_term_config=self.memory_config['long_term'],
            device=device
        )
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Training Loop'da Kullanım Örneği

```python
# Training loop example with UBÖO
model = MMRecModel(vocab_size=32000, use_uboo=True, lambda_P=0.1)

for batch in dataloader:
    input_ids = batch['input_ids']
    targets = batch['labels']
    
    # Forward pass with auxiliary loss
    logits, L_Aux_total = model(input_ids, return_auxiliary_loss=True)
    
    # Main loss (language modeling loss)
    L_main = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-1
    )
    
    # CRITICAL: Total loss = Main loss + Scaled auxiliary loss
    # L_total = L_main + λ_P * Σ_{l=1}^{L} L_Aux^{(l)}
    L_total = L_main + L_Aux_total
    
    # Backward pass
    L_total.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    print(f"L_main: {L_main.item():.4f}, L_Aux: {L_Aux_total.item():.4f}, L_total: {L_total.item():.4f}")
```

---

## Özet

### 1. Gradyan İzolasyonu (`mdi.py`)

**Kritik Kod Parçaları**:
- **Detach Operation**: `M_updated_detached = M_updated.detach()`
- **Planning Error Computation**: `P_error = W_planning_error(M_updated_detached) - W_planning_target(M_target)`
- **Auxiliary Loss**: `L_Aux = mean(||P_error||²)`

**Önemli Detaylar**:
- `detach()` gradyan akışını keser
- Planning error hesaplaması M_updated'den bağımsızdır
- Unbiased backpropagation sağlanır

### 2. Auxiliary Loss Toplama (`model.py`)

**Kritik Kod Parçaları**:
- **Loss Collection**: Tüm katmanlardan L_Aux toplama
- **Loss Summation**: `L_Aux_sum = sum(auxiliary_losses)`
- **Scaling**: `L_Aux_total = lambda_P * L_Aux_sum`
- **Total Loss**: `L_total = L_main + L_Aux_total`

**Önemli Detaylar**:
- Her katmanın L_Aux'u toplanır
- Lambda_P faktörü ile ölçeklenir (tipik: 0.01-0.1)
- Ana kayıp çıktısına eklenir

### Kritik Notlar

1. **Gradient Isolation**: `detach()` ile gradyan akışı kesilir
2. **Unbiased Backpropagation**: Error orthogonalization ile sağlanır
3. **Auxiliary Loss Scaling**: Lambda_P faktörü ile kontrol edilir
4. **Layer-wise Collection**: Her katmanın L_Aux'u toplanır

---

**Doküman Versiyonu**: 1.0  
**Oluşturulma Tarihi**: 2025-01-27  
**Amaç**: UBÖO (Mekanizma 3) entegrasyonu için kritik kod parçaları


