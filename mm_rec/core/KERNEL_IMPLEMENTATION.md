# Work-Efficient Parallel Scan Kernel Implementation

## Overview

This document describes the work-efficient parallel scan kernel implementation for MM-Rec's associative scan (exponential product) operation.

## Algorithm: Blelloch Scan

The kernel implements the **work-efficient parallel scan** algorithm (Blelloch, 1990) with the following phases:

### 1. Up-Sweep Phase (Reduction Tree)
- **Purpose**: Build a reduction tree to compute the total sum of the block
- **Complexity**: O(log n) depth, O(n) work
- **Operation**: Log-space addition using stable log-sum-exp
- **Result**: Last element contains the block's total sum

### 2. Down-Sweep Phase (Prefix Propagation)
- **Purpose**: Propagate prefixes from root to leaves
- **Complexity**: O(log n) depth, O(n) work
- **Operation**: Combine prefixes with subtree totals
- **Result**: Each position gets its cumulative prefix sum

### 3. Block-to-Block Carry-Over
- **Purpose**: Propagate prefix sums across blocks
- **Mechanism**: Each block computes its total prefix and passes it to the next block
- **Implementation**: Python-level loop with carry buffers

## Kernel Signature

```python
@triton.jit
def associative_scan_parallel_kernel(
    input_ptr,          # [BATCH, HEADS, SEQ_LEN, D_HEAD]
    output_ptr,         # [BATCH, HEADS, SEQ_LEN, D_HEAD]
    carry_in_ptr,       # [BATCH, HEADS, D_HEAD] - prefix from previous block
    carry_out_ptr,      # [BATCH, HEADS, D_HEAD] - prefix for next block
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    stride_batch,
    stride_heads,
    stride_seq,
    stride_dim,
    block_idx,          # Current block index
    has_carry_in: tl.constexpr,
    has_carry_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
```

## Key Features

### 1. Log-Sum-Exp Stability
```python
def stable_log_sum_exp(a, b):
    max_val = maximum(a, b)
    diff = abs(a - b)
    diff_clamped = minimum(diff, 20.0)
    return max_val + log1p(exp(-diff_clamped))
```

### 2. Block Processing
- Each block processes `BLOCK_SIZE` elements (default: 512-1024)
- Blocks are processed sequentially with carry-over propagation
- Multiple blocks can run in parallel for different (batch, head, dim) combinations

### 3. Memory Layout
- **Input**: `[BATCH, HEADS, SEQ_LEN, D_HEAD]` - log-space values
- **Output**: `[BATCH, HEADS, SEQ_LEN, D_HEAD]` - cumulative log-sums
- **Carry buffers**: `[BATCH, HEADS, D_HEAD]` - block prefixes

## Performance Characteristics

### Complexity
- **Time**: O(log n) parallel depth per block
- **Work**: O(n) total operations
- **Blocks**: Sequential processing for carry-over (O(num_blocks))

### Scalability
- Supports sequences up to 32K+ tokens
- Block size optimized for GPU warp/wavefront size
- Memory coalescing for efficient GPU access

### Numerical Stability
- All operations in log-space using FP32 precision
- Clamping to [-50, 0] range prevents underflow/overflow
- Stable log-sum-exp prevents numerical errors

## Usage

```python
from mm_rec.core import associative_scan_exponential

# Input: [BATCH, HEADS, SEQ_LEN, D_HEAD]
gamma = torch.rand(2, 8, 32768, 128, dtype=torch.bfloat16, device='cuda')

# Compute cumulative product
result = associative_scan_exponential(gamma)
```

## Implementation Notes

1. **Block Size Selection**: 
   - For seq_len >= 1024: BLOCK_SIZE = 1024
   - For seq_len >= 512: BLOCK_SIZE = 512
   - For seq_len >= 256: BLOCK_SIZE = 256
   - Default: BLOCK_SIZE = 128

2. **Carry-Over Propagation**:
   - First block: no carry-in (identity: 0.0 in log-space)
   - Intermediate blocks: add carry-in prefix to all positions
   - Last block: no carry-out (not needed)

3. **Grid Launch**:
   - Grid dimensions: `(batch_size, num_heads, head_dim)`
   - Each thread block processes one (batch, head, dim) combination
   - Blocks are processed sequentially for carry-over

## Testing

The implementation includes correctness tests comparing against sequential `torch.cumprod()`:

```python
from mm_rec.core.associative_scan_triton import test_associative_scan_correctness

test_associative_scan_correctness()
```

Expected tolerance: `max_diff < 1e-3` for BF16 inputs.

