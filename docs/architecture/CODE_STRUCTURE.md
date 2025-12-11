# MM-Rec Code Structure & API Design
## Direct Implementation Guide

---

## PROJECT STRUCTURE

```
mm-rec/
├── mm_rec/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── associative_scan.py      # Associative scan implementation
│   │   ├── hds.py                    # Hierarchical data structure
│   │   ├── mdi.py                    # Memory decay/integration
│   │   └── memory_state.py           # Memory state management
│   ├── blocks/
│   │   ├── __init__.py
│   │   ├── mm_rec_block.py           # Main MM-Rec block
│   │   └── attention.py               # Multi-memory attention
│   ├── cuda/
│   │   ├── __init__.py
│   │   ├── scan_kernel.cu            # CUDA kernel for scan
│   │   ├── scan_kernel.h             # Kernel headers
│   │   └── compile.py                # CUDA compilation utilities
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── fsdp_wrapper.py           # FSDP integration
│   │   └── sequence_parallel.py      # Sequence parallelism
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── profiling.py              # Performance profiling
│   │   └── debugging.py              # Debugging utilities
│   └── config.py                     # Configuration management
├── tests/
│   ├── test_associative_scan.py
│   ├── test_hds.py
│   ├── test_mdi.py
│   ├── test_mm_rec_block.py
│   └── test_gradients.py
├── training/
│   ├── train.py                      # Main training script
│   └── configs/
│       ├── mmrec_7b.yaml
│       └── training_config.yaml
└── requirements.txt
```

---

## CORE API SPECIFICATIONS

### 1. Associative Scan API

```python
# mm_rec/core/associative_scan.py

import torch
from torch.autograd import Function

class AssociativeScanExponential(Function):
    """
    Parallel associative scan for cumulative exponential product.
    Computes: y_t = ∏ᵢ₌₁ᵗ γᵢ using Log-Sum-Exp for numerical stability.
    
    Args:
        gamma: [batch, num_heads, seq_len, dim] decay coefficients γ
    
    Returns:
        output: [batch, num_heads, seq_len, dim] cumulative products
    """
    @staticmethod
    def forward(ctx, gamma):
        """
        Compute cumulative exponential product in log-space for stability.
        """
        # Convert to log-space with clamping
        log_gamma = torch.clamp(torch.log(gamma + 1e-8), min=-50.0, max=0.0)
        
        # Call CUDA kernel for log-space scan (sum in log-space)
        log_cumsum = associative_scan_log_kernel(log_gamma, operator='add')
        
        # Convert back to linear space with stability
        max_log = torch.max(log_cumsum, dim=-1, keepdim=True)[0]
        stable_log = log_cumsum - max_log
        cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
        
        ctx.save_for_backward(gamma, cumulative_product, log_cumsum)
        return cumulative_product
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient computation for exponential product
        gamma, cumulative_product, log_cumsum = ctx.saved_tensors
        grad_gamma = cumulative_product * grad_output
        return grad_gamma

def associative_scan_exponential(gamma: torch.Tensor) -> torch.Tensor:
    """
    User-facing function for exponential product scan.
    Computes cumulative product: ∏ᵢ₌₁ᵗ γᵢ
    """
    return AssociativeScanExponential.apply(gamma)
```

### 2. HDS API

```python
# mm_rec/core/hds.py

import torch
import torch.nn as nn
from typing import List, Tuple

class HDSHierarchy:
    """
    Hierarchical Data Structure for memory organization.
    """
    def __init__(
        self,
        num_levels: int = 3,
        chunk_size: int = 128,
        mem_dim: int = 512
    ):
        self.num_levels = num_levels
        self.chunk_size = chunk_size
        self.mem_dim = mem_dim
    
    def build_hierarchy(
        self,
        memories: torch.Tensor  # [B, T, N, D]
    ) -> List[torch.Tensor]:
        """
        Build hierarchical memory structure.
        
        Returns:
            List of tensors, one per hierarchy level
        """
        pass
    
    def query(
        self,
        query: torch.Tensor,  # [B, T, D]
        hierarchy: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Query across hierarchy levels.
        """
        pass

class HDSAttention(nn.Module):
    """
    Attention mechanism over HDS hierarchy.
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels)
        ])
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        hierarchy: List[torch.Tensor]
    ) -> torch.Tensor:
        # Multi-level attention implementation
        pass
```

### 3. MDI API

```python
# mm_rec/core/mdi.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDecay(nn.Module):
    """
    Learnable decay coefficients for memory integration.
    """
    def __init__(
        self,
        num_memories: int,
        hidden_dim: int,
        decay_init: float = 0.99
    ):
        super().__init__()
        self.num_memories = num_memories
        
        # Base decay parameter (per memory bank)
        decay_logit = self._logit(decay_init)
        self.decay_logits = nn.Parameter(
            torch.ones(num_memories) * decay_logit
        )
        
        # Context-dependent modulation
        self.decay_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_memories),
            nn.Sigmoid()
        )
    
    def _logit(self, p: float) -> float:
        """Convert probability to logit."""
        return torch.log(torch.tensor(p / (1 - p)))
    
    def forward(
        self,
        memory_old: torch.Tensor,  # [B, T, N, D]
        memory_new: torch.Tensor,  # [B, T, N, D]
        context: torch.Tensor      # [B, T, D]
    ) -> torch.Tensor:
        """
        Compute decay coefficients and apply memory update.
        """
        # Base decay
        base_decay = torch.sigmoid(self.decay_logits)  # [N]
        base_decay = base_decay.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, 1]
        
        # Context modulation
        modulation = self.decay_modulation(context)  # [B, T, N]
        modulation = modulation.unsqueeze(-1)  # [B, T, N, 1]
        
        # Final decay
        decay = base_decay * modulation
        decay = torch.clamp(decay, min=1e-6, max=1.0 - 1e-6)
        
        # Memory update
        updated = decay * memory_old + (1 - decay) * memory_new
        
        return updated

class MemoryIntegration(nn.Module):
    """
    Gated memory integration with residual connection.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        old_memory: torch.Tensor,
        new_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate new memory into old with gating.
        """
        gate_value = self.gate(torch.cat([old_memory, new_memory], dim=-1))
        integrated = gate_value * new_memory + (1 - gate_value) * old_memory
        return integrated + old_memory  # Residual connection
```

### 4. Memory State API

```python
# mm_rec/core/memory_state.py

import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MemoryBank:
    """
    Single memory bank state.
    """
    k: torch.Tensor  # [B, T, H, D] - Key states
    v: torch.Tensor  # [B, T, H, D] - Value states
    state: torch.Tensor  # [B, T, D] - Aggregated state
    decay_coeff: torch.Tensor  # [B, T] - Current decay coefficients (γ)
    hidden_states: torch.Tensor  # [B, T, D] - Short-term memory h_t

class MemoryState:
    """
    Manages all memory banks for MM-Rec block.
    Includes both short-term (h_t) and long-term (M) memory.
    """
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_memories: int,
        num_heads: int,
        head_dim: int,
        mem_dim: int,
        memory_size_M: int,  # Long-term memory size
        device: torch.device
    ):
        self.num_memories = num_memories
        self.memory_size_M = memory_size_M
        self.banks: List[MemoryBank] = []
        
        # Short-term memory (h_t)
        self.hidden_states = torch.zeros(batch_size, seq_len, mem_dim, device=device)
        
        # Long-term memory (M) - O(M) access cost
        self.long_term_memory = torch.zeros(
            batch_size, num_memories, memory_size_M, mem_dim, device=device
        )
        
        for _ in range(num_memories):
            bank = MemoryBank(
                k=torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device),
                v=torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device),
                state=torch.zeros(batch_size, seq_len, mem_dim, device=device),
                decay_coeff=torch.ones(batch_size, seq_len, device=device) * 0.99,
                hidden_states=torch.zeros(batch_size, seq_len, mem_dim, device=device)
            )
            self.banks.append(bank)
    
    def update_bank(
        self,
        bank_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        new_state: torch.Tensor,
        decay_coeff: torch.Tensor
    ):
        """Update a specific memory bank."""
        self.banks[bank_idx].k = new_k
        self.banks[bank_idx].v = new_v
        self.banks[bank_idx].state = new_state
        self.banks[bank_idx].decay_coeff = decay_coeff
    
    def get_all_states(self) -> torch.Tensor:
        """Concatenate all memory states."""
        states = [bank.state for bank in self.banks]
        return torch.cat(states, dim=-1)  # [B, T, N*D]
    
    def reset(self):
        """Reset all memory banks to zero."""
        for bank in self.banks:
            bank.k.zero_()
            bank.v.zero_()
            bank.state.zero_()
            bank.decay_coeff.fill_(0.99)
```

### 5. MM-Rec Block API

```python
# mm_rec/blocks/mm_rec_block.py

import torch
import torch.nn as nn
from typing import Optional

from ..core.associative_scan import associative_scan
from ..core.hds import HDSHierarchy, HDSAttention
from ..core.mdi import LearnableDecay, MemoryIntegration
from ..core.memory_state import MemoryState

class MMRecBlock(nn.Module):
    """
    Complete MM-Rec block implementation.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_memories: int = 8,
        mem_dim: int = 512,
        num_heads: int = 32,
        num_hds_levels: int = 3,
        chunk_size: int = 128,
        ffn_dim: int = 11008,
        dropout: float = 0.1,
        decay_init: float = 0.99
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_memories = num_memories
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Input projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, hidden_dim)  # For z_t in core formula
        
        # Gating projection for core formula: σ(W_g h_{t-1})
        self.gating_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Multi-memory projections
        self.mem_k_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_memories)
        ])
        self.mem_v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_memories)
        ])
        
        # HDS components
        self.hds_hierarchy = HDSHierarchy(
            num_levels=num_hds_levels,
            chunk_size=chunk_size,
            mem_dim=mem_dim
        )
        self.hds_attention = HDSAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_hds_levels
        )
        
        # MDI components
        self.learnable_decay = LearnableDecay(
            num_memories=num_memories,
            hidden_dim=hidden_dim,
            decay_init=decay_init
        )
        self.memory_integration = MemoryIntegration(hidden_dim=mem_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_memories * mem_dim, hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Normalization
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        memory_state: Optional[MemoryState] = None
    ) -> tuple[torch.Tensor, MemoryState]:
        """
        Forward pass through MM-Rec block.
        
        Returns:
            output: [B, T, D] transformed tensor
            memory_state: Updated memory state
        """
        residual = x
        x = self.norm1(x)
        
        # 1. Input Projection
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        z = self.z_proj(x)  # [B, T, D] - For gated update z_t
        
        # Reshape for multi-head
        B, T, D = x.shape
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # 2. Core Recurrence Formula: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
        if memory_state is not None:
            h_prev = memory_state.hidden_states  # [B, T, D]
            gate_signal = torch.sigmoid(self.gating_proj(h_prev))  # σ(W_g h_{t-1})
        else:
            h_prev = torch.zeros_like(x)
            gate_signal = torch.ones_like(x) * 0.5
        
        # Get decay coefficients γ
        gamma = self.learnable_decay.decay_logits.sigmoid()  # [num_memories]
        gamma = gamma.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, 1]
        
        # Compute cumulative product ∏ᵢ γᵢ via associative scan (in log-space)
        from ..core.associative_scan import associative_scan_exponential
        gamma_cumprod = associative_scan_exponential(gamma)  # [1, 1, T, N, 1]
        
        # Apply core formula
        h_t = z * gate_signal + gamma_cumprod.squeeze() * h_prev
        
        # 3. Multi-Memory Attention (query h_t against long-term memory M)
        memory_attentions = []
        for i in range(self.num_memories):
            if memory_state is not None:
                # Query h_t against long-term memory M (O(M) access)
                mem_k = memory_state.long_term_memory[:, i, :, :]  # [B, M, D]
                mem_v = memory_state.long_term_memory[:, i, :, :]
            else:
                mem_k = k.mean(dim=2)  # [B, T, D] - fallback
                mem_v = v.mean(dim=2)
            
            # Attention computation
            scores = torch.einsum('bthd,bthd->bth', q, mem_k) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.einsum('bth,bthd->bthd', attn_weights, mem_v)
            memory_attentions.append(attn_output)
        
        memory_attentions = torch.stack(memory_attentions, dim=2)  # [B, T, N, H, D]
        
        # Note: Associative scan already applied for gamma_cumprod above
        # Memory attentions are already computed
        memory_attentions = torch.stack(memory_attentions, dim=2)  # [B, T, N, H, D]
        scanned = memory_attentions.view(B, T, self.num_memories, -1)  # [B, T, N, H*D]
        
        # 4. HDS Aggregation (include long-term memory M)
        hds_hierarchy = self.hds_hierarchy.build_hierarchy(scanned)
        if memory_state is not None:
            hds_hierarchy.append(memory_state.long_term_memory)  # Add M level
        
        # 5. HDS Query (using h_t)
        hds_output = self.hds_attention(h_t, hds_hierarchy)
        
        # 6. MDI Update (update both h_t and long-term memory M)
        if memory_state is None:
            memory_state = MemoryState(
                batch_size=B,
                seq_len=T,
                num_memories=self.num_memories,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mem_dim=self.mem_dim,
                memory_size_M=1024,  # Configurable
                device=x.device
            )
        
        # Update short-term memory h_t
        updated_h_t = h_t  # Already computed via core formula
        
        # Update long-term memory M (O(M) operation)
        for i in range(self.num_memories):
            # Incremental update to M based on h_t
            memory_state.long_term_memory[:, i, :, :] = (
                0.99 * memory_state.long_term_memory[:, i, :, :] +
                0.01 * h_t.mean(dim=1, keepdim=True).expand(-1, memory_state.memory_size_M, -1)
            )
            
            # Update memory bank states
            old_mem = memory_state.banks[i].state
            new_mem = hds_output[:, :, i * self.mem_dim:(i+1) * self.mem_dim]
            
            # Decay and integration
            decayed = self.learnable_decay(old_mem, new_mem, x)
            integrated = self.memory_integration(old_mem, decayed)
            
            # Update memory state
            memory_state.update_bank(
                i,
                new_k=k,
                new_v=v,
                new_state=integrated,
                decay_coeff=self.learnable_decay.decay_logits.sigmoid()
            )
        
        # Update hidden states
        memory_state.hidden_states = updated_h_t
        
        # 7. Output Projection
        memory_states_concat = memory_state.get_all_states()  # [B, T, N*D]
        output = self.out_proj(torch.cat([updated_h_t, memory_states_concat], dim=-1))
        
        # Residual connection
        output = output + residual
        
        # FFN
        residual = output
        output = self.norm2(output)
        output = self.ffn(output)
        output = output + residual
        
        return output, memory_state
```

### 6. Model API

```python
# mm_rec/model.py

import torch
import torch.nn as nn
from .blocks.mm_rec_block import MMRecBlock
from .core.memory_state import MemoryState

class MMRecModel(nn.Module):
    """
    Complete MM-Rec model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # MM-Rec blocks (24 layers as per spec)
        self.blocks = nn.ModuleList([
            MMRecBlock(
                hidden_dim=config.hidden_dim,
                num_memories=config.num_memories,
                mem_dim=config.mem_dim,
                num_heads=config.num_heads,
                num_hds_levels=config.num_hds_levels,
                chunk_size=config.chunk_size,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                decay_init=config.decay_init
            )
            for _ in range(config.num_layers)  # num_layers = 24 (REQUIRED)
        ])
        
        # Output
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None
    ):
        x = self.embedding(input_ids)
        
        if memory_states is None:
            memory_states = [None] * len(self.blocks)
        
        new_memory_states = []
        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem_state = block(x, mem_state)
            new_memory_states.append(new_mem_state)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_memory_states
```

---

## CUDA KERNEL TEMPLATE

```cuda
// mm_rec/cuda/scan_kernel.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__device__ T scan_operator_log(T log_a, T log_b) {
    // Log-space addition for exponential product
    // log(a * b) = log(a) + log(b)
    // Use stable log-sum-exp: max(log_a, log_b) + log(1 + exp(-abs(log_a - log_b)))
    T max_val = fmaxf(log_a, log_b);
    T diff = fabsf(log_a - log_b);
    T stable_log_sum = max_val + log1pf(expf(-diff));
    return stable_log_sum;
}

template<typename T>
__global__ void associative_scan_up_sweep(
    const T* input,
    T* output,
    int seq_len,
    int dim
) {
    // Up-sweep phase implementation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... implementation
}

template<typename T>
__global__ void associative_scan_down_sweep(
    const T* input,
    T* output,
    int seq_len,
    int dim
) {
    // Down-sweep phase implementation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... implementation
}

// Python bindings via pybind11 or torch extension
```

---

## USAGE EXAMPLE

```python
import torch
from mm_rec.model import MMRecModel
from mm_rec.config import MMREC_7B_CONFIG

# Initialize model
config = MMREC_7B_CONFIG
model = MMRecModel(config).cuda()

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 2048)).cuda()
logits, memory_states = model(input_ids)

# Training step
loss = compute_loss(logits, targets)
loss.backward()
optimizer.step()
```

---

## IMPLEMENTATION PRIORITY

1. **Week 1**: Associative scan (CPU fallback first, then CUDA)
2. **Week 2**: HDS hierarchy construction and query
3. **Week 3**: MDI decay mechanism
4. **Week 4**: Complete MM-Rec block integration
5. **Week 5**: CUDA kernel optimization
6. **Week 6**: Distributed training integration
7. **Week 7**: Testing and validation
8. **Week 8**: Performance tuning and benchmarking

