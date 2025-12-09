# C++ Optimization Plan for MM-Rec

## Performance Critical Components

### 1. Associative Scan (Exponential Product)
**Current**: Triton kernels
**C++ Optimization**: CUDA C++ kernels for better control

**Benefits**:
- Better memory access patterns
- Custom optimization for specific hardware
- Lower-level control
- Potentially faster than Triton

**Files to optimize**:
- `mm_rec/core/associative_scan_triton.py` → `mm_rec/cuda/associative_scan.cu`

### 2. Sequential Processing Loop
**Current**: Python `for t in range(seq_len):` loop
**C++ Optimization**: C++ extension with Cython or PyTorch C++ API

**Problem**: 
- CPU-GPU synchronization overhead
- Python loop overhead
- Memory allocation in loop

**Files to optimize**:
- `mm_rec/blocks/mm_rec_block.py` → `mm_rec/cpp/mm_rec_block_cpp.cpp`

### 3. Memory State Updates
**Current**: Python tensor operations
**C++ Optimization**: Custom CUDA kernels

**Files to optimize**:
- `mm_rec/core/memory_state.py` → `mm_rec/cuda/memory_state.cu`

### 4. MDI (Memory Decay Integration)
**Current**: PyTorch operations
**C++ Optimization**: Fused CUDA kernel

**Files to optimize**:
- `mm_rec/core/mdi.py` → `mm_rec/cuda/mdi.cu`

### 5. HDS Query Operations
**Current**: Python tensor operations
**C++ Optimization**: CUDA kernels for tree traversal

**Files to optimize**:
- `mm_rec/core/hds.py` → `mm_rec/cuda/hds.cu`

---

## Implementation Strategy

### Option 1: PyTorch C++ Extensions (Recommended)

**Advantages**:
- Native PyTorch integration
- Automatic gradient support
- Easy to use from Python
- Good documentation

**Structure**:
```
mm_rec/
├── cpp/
│   ├── associative_scan.cpp
│   ├── mm_rec_block.cpp
│   ├── memory_state.cpp
│   └── setup.py
└── cuda/
    ├── associative_scan_kernel.cu
    ├── memory_state_kernel.cu
    └── mdi_kernel.cu
```

### Option 2: Cython

**Advantages**:
- Easy Python-like syntax
- Good performance
- Easy integration

**Disadvantages**:
- Less control than C++
- CUDA support more complex

### Option 3: Pure CUDA with ctypes

**Advantages**:
- Maximum control
- Best performance potential

**Disadvantages**:
- More complex
- Manual memory management
- No automatic gradients

---

## Priority List

### High Priority (Biggest Impact)

1. **Sequential Processing Loop** (mm_rec_block.py)
   - **Impact**: 10-100x speedup possible
   - **Current**: Python loop with CPU-GPU sync
   - **Optimization**: C++ loop with fused operations
   - **Estimated speedup**: 20-50x

2. **Associative Scan Kernel**
   - **Impact**: 2-5x speedup
   - **Current**: Triton (good but can be better)
   - **Optimization**: CUDA C++ with custom optimizations
   - **Estimated speedup**: 2-3x

### Medium Priority

3. **MDI Operations**
   - **Impact**: 2-3x speedup
   - **Current**: Multiple PyTorch ops
   - **Optimization**: Fused CUDA kernel
   - **Estimated speedup**: 2-3x

4. **Memory State Updates**
   - **Impact**: 1.5-2x speedup
   - **Current**: Python tensor ops
   - **Optimization**: CUDA kernels
   - **Estimated speedup**: 1.5-2x

### Low Priority

5. **HDS Query**
   - **Impact**: 1.2-1.5x speedup
   - **Current**: Already efficient
   - **Optimization**: Minor improvements
   - **Estimated speedup**: 1.2-1.5x

---

## Implementation Example: Sequential Loop Optimization

### Current Python Code
```python
# mm_rec/blocks/mm_rec_block.py
for t in range(seq_len):
    x_t = x[:, t:t+1, :]
    h_prev = state.get_state_at_step('short', t-1)
    # ... operations ...
    state.update_state_sequential('short', h_t, t)
```

### C++ Extension
```cpp
// mm_rec/cpp/mm_rec_block_cpp.cpp
#include <torch/extension.h>
#include <vector>

torch::Tensor mm_rec_block_forward_cpp(
    torch::Tensor x,
    torch::Tensor state_h,
    // ... other parameters
) {
    // Fused C++ loop
    // No Python overhead
    // Better memory management
    // ...
}
```

### Python Wrapper
```python
# mm_rec/blocks/mm_rec_block.py
import mm_rec_cpp

def forward(self, x, state, ...):
    if self.use_cpp_optimization:
        return mm_rec_cpp.mm_rec_block_forward_cpp(x, state, ...)
    else:
        # Fallback to Python
        ...
```

---

## Setup and Build

### 1. Setup.py for C++ Extensions
```python
# mm_rec/cpp/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mm_rec_cpp',
    ext_modules=[
        CUDAExtension(
            'mm_rec_cpp',
            [
                'associative_scan.cpp',
                'mm_rec_block.cpp',
                'cuda/associative_scan_kernel.cu',
                'cuda/mm_rec_block_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_75']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 2. Build Command
```bash
cd mm_rec/cpp
python setup.py build_ext --inplace
```

### 3. Installation
```bash
pip install -e .
```

---

## Performance Benchmarks

### Expected Improvements

| Component | Current | C++ Optimized | Speedup |
|-----------|---------|---------------|---------|
| Sequential Loop | 70s/step | 2-5s/step | 14-35x |
| Associative Scan | Baseline | Optimized | 2-3x |
| MDI Operations | Baseline | Fused | 2-3x |
| Memory Updates | Baseline | Optimized | 1.5-2x |
| **Overall** | **70s/step** | **2-5s/step** | **14-35x** |

### Real-World Impact

**Before C++ optimization**:
- Training step: ~70 seconds (CPU)
- 100 steps: ~2 hours

**After C++ optimization**:
- Training step: ~2-5 seconds (CPU) or ~0.1-0.5s (GPU)
- 100 steps: ~5-10 minutes (CPU) or ~10-50 seconds (GPU)

---

## Implementation Steps

### Phase 1: Setup (1 day)
1. Create C++ extension structure
2. Setup build system
3. Create basic Python-C++ bridge

### Phase 2: Sequential Loop (2-3 days)
1. Port sequential loop to C++
2. Fuse operations
3. Optimize memory access
4. Test and benchmark

### Phase 3: Associative Scan (2-3 days)
1. Port Triton kernel to CUDA C++
2. Optimize memory access patterns
3. Add custom optimizations
4. Test and benchmark

### Phase 4: Other Components (1-2 days each)
1. MDI operations
2. Memory state updates
3. HDS queries

### Phase 5: Integration and Testing (2-3 days)
1. Integrate all components
2. End-to-end testing
3. Performance benchmarking
4. Documentation

**Total Estimated Time**: 2-3 weeks

---

## Code Examples

### Example 1: Sequential Loop C++ Extension

```cpp
// mm_rec/cpp/mm_rec_block_cpp.cpp
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mm_rec_block_forward_sequential(
    torch::Tensor x,           // [batch, seq_len, hidden_dim]
    torch::Tensor state_h,     // [batch, hidden_dim]
    torch::Tensor W_q,
    torch::Tensor W_k,
    torch::Tensor W_v,
    torch::Tensor W_z,
    // ... other weights
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_dim = x.size(2);
    
    // Pre-allocate output
    auto output = torch::zeros({batch_size, seq_len, hidden_dim}, x.options());
    
    // Fused C++ loop - no Python overhead
    for (int t = 0; t < seq_len; ++t) {
        // Get input slice
        auto x_t = x.slice(1, t, t+1).squeeze(1);  // [batch, hidden_dim]
        
        // All operations fused in C++
        // QKV projections
        auto q_t = torch::matmul(x_t, W_q);
        auto k_t = torch::matmul(x_t, W_k);
        auto v_t = torch::matmul(x_t, W_v);
        auto z_t = torch::matmul(x_t, W_z);
        
        // MDI operations
        // ... fused operations ...
        
        // Update state
        state_h = h_t;  // In-place update
        
        // Store output
        output.slice(1, t, t+1) = h_t.unsqueeze(1);
    }
    
    return {output, state_h};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_rec_block_forward_sequential", 
          &mm_rec_block_forward_sequential,
          "MM-Rec Block forward (C++ optimized)");
}
```

### Example 2: CUDA Kernel for Associative Scan

```cpp
// mm_rec/cuda/associative_scan_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void associative_scan_parallel_kernel(
    const float* log_gamma,      // Input: log(gamma)
    float* log_cumsum,           // Output: log(cumulative product)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Work-efficient parallel scan (Blelloch algorithm)
    // Optimized CUDA implementation
    // ...
}

torch::Tensor associative_scan_cuda(
    torch::Tensor log_gamma
) {
    // Launch CUDA kernel
    // ...
}
```

---

## Dependencies

### Required
- **PyTorch**: C++ extensions support
- **CUDA**: For GPU kernels
- **C++17**: Modern C++ features
- **nvcc**: NVIDIA CUDA compiler

### Optional
- **Cython**: Alternative to PyTorch C++ extensions
- **Ninja**: Faster builds
- **CMake**: Alternative build system

---

## Testing Strategy

### Unit Tests
- Compare C++ output with Python reference
- Test gradient correctness
- Test edge cases

### Performance Tests
- Benchmark each component
- Compare before/after
- Profile with NVIDIA Nsight

### Integration Tests
- End-to-end training
- Checkpoint compatibility
- Multi-GPU support

---

## Migration Path

### Step 1: Add C++ Extension (Non-Breaking)
- Keep Python implementation as fallback
- Add `use_cpp_optimization` flag
- Test both paths

### Step 2: Optimize Critical Paths
- Start with sequential loop
- Then associative scan
- Then other components

### Step 3: Make C++ Default
- After thorough testing
- Keep Python as fallback
- Document performance gains

---

## Resources

- **PyTorch C++ Extensions**: https://pytorch.org/tutorials/advanced/cpp_extension.html
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **PyTorch C++ API**: https://pytorch.org/cppdocs/
- **NVIDIA Nsight**: Performance profiling

---

## Conclusion

C++ optimization can provide **10-50x speedup** for MM-Rec, especially for:
1. Sequential processing loop (biggest impact)
2. Associative scan operations
3. Fused operations (MDI, memory updates)

**Recommended approach**: Start with PyTorch C++ extensions for sequential loop, then optimize other components.

