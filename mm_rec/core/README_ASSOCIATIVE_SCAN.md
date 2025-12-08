# Associative Scan Exponential - Triton Implementation

## Overview

This module implements the **Associative Scan (Exponential Product)** operation for MM-Rec architecture using Triton kernels. The operation computes cumulative exponential products with numerical stability using the **Log-Sum-Exp** pattern.

## Mathematical Definition

**Operation**: Compute cumulative exponential product
```
Y_t = ∏_{i=1}^t γ_i
```

**Stability Pattern**: Log-Sum-Exp
```
1. L_i = log(γ_i + ε)                    # Convert to log-space
2. L_i = clamp(L_i, -50.0, 0.0)          # Clamp for stability
3. L_sum_t = Σ_{i=1}^t L_i               # Cumulative sum in log-space
4. Y_t = exp(L_sum_t)                    # Convert back to linear space
```

## Usage

### Basic Usage

```python
import torch
from mm_rec.core import associative_scan_exponential

# Create input tensor [BATCH, HEADS, SEQ_LEN, D_HEAD]
batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 128
gamma = torch.rand(batch_size, num_heads, seq_len, head_dim, 
                   dtype=torch.bfloat16, device='cuda')
gamma = gamma * 0.9 + 0.05  # Ensure values in [0.05, 0.95]

# Compute cumulative product
result = associative_scan_exponential(gamma)
# result.shape: [2, 8, 1024, 128]
```

### Integration with MM-Rec Block

```python
from mm_rec.core import associative_scan_exponential

class MMRecBlock(nn.Module):
    def forward(self, x, memory_state):
        # ... other operations ...
        
        # Get decay coefficients
        gamma = self.get_decay_coefficients()  # [B, H, T, D]
        
        # Compute cumulative product via associative scan
        gamma_cumprod = associative_scan_exponential(gamma)
        
        # Use in core recurrence formula
        h_t = z_t * gate_signal + gamma_cumprod * h_prev
        
        # ... rest of forward pass ...
```

## Implementation Details

### Precision Handling

- **Input/Output**: BF16 (bfloat16) for memory efficiency
- **Internal Computation**: FP32 for numerical stability
  - Log operations: FP32
  - Exponential operations: FP32
  - Final conversion: Back to BF16

### Numerical Stability

1. **Epsilon Addition**: `log(gamma + 1e-8)` prevents `log(0)`
2. **Log Clamping**: `clamp(log(gamma), -50.0, 0.0)` prevents:
   - Underflow: `exp(-50) ≈ 0`
   - Overflow: `exp(0) = 1`
3. **Stable Exponential**: `exp(log_sum - max) * exp(max)` pattern
4. **Stable Log-Sum-Exp**: `max(a, b) + log1p(exp(-abs(a - b)))`

### Performance

- **Complexity**: O(log n) depth for parallel scan
- **Block Size**: Configurable (default: 256 for seq_len ≥ 256)
- **Memory**: Efficient shared memory usage
- **Scalability**: Supports sequences up to 32K+ tokens

## Testing

```python
from mm_rec.core.associative_scan_triton import test_associative_scan_correctness

# Run correctness test
if torch.cuda.is_available():
    test_associative_scan_correctness()
```

## Requirements

- PyTorch 2.0+
- Triton 2.0+
- CUDA-capable GPU (compute capability 7.0+)
- CUDA Toolkit 11.8+

## Limitations and Future Work

1. **Cross-Block Accumulation**: Current implementation processes blocks independently. For very long sequences, cross-block accumulation may be needed.

2. **Optimization**: Current kernel uses sequential scan within blocks. Can be optimized to use work-efficient parallel scan (Blelloch scan).

3. **Memory Access**: Can be further optimized for better memory coalescing patterns.

## References

- Triton Documentation: https://triton-lang.org/
- Parallel Scan Algorithms: Blelloch (1990)
- Numerical Stability: Higham (2002) - "Accuracy and Stability of Numerical Algorithms"

