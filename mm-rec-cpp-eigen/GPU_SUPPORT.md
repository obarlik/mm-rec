# MM-Rec GPU Integration Strategy

## Current Status
The current project (`mm-rec-cpp-eigen`) is highly optimized for CPU using **Intel MKL** and **OpenMP**. It does not currently contain active CUDA/HIP code.

## Path to GPU Readiness
To support GPU without disrupting the current CPU-optimized workflow, we will use a **Hybrid Backend** approach using CMake switches.

### 1. CMake Configuration (`cmake/gpu.cmake`)
We have prepared a CMake module to detect and enable CUDA.
**Usage:** To enable GPU, run cmake with `-DUSE_CUDA=ON`.

### 2. Abstraction Layer
The `Tensor` class currently uses `Eigen::Matrix` on CPU. For GPU, we should introduce a backend switch:

```cpp
#ifdef USE_CUDA
    // Delegate to cuBLAS / CUDA Kernels
    void matmul_gpu(const Tensor& a, const Tensor& b, Tensor& out);
#else
    // Current Eigen/MKL CPU implementation
    void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& out);
#endif
```

### 3. Kernel Migration Plan
| Component | CPU Implementation | GPU Implementation Plan |
|-----------|--------------------|------------|
| Matmul | Eigen + MKL (AVX512) | cuBLAS |
| Element-wise | OpenMP SIMD | CUDA Custom Kernels |
| Normalization | Manual Loop + AVX | CUDA Kernel (Thrust/CUB) |
| MoE Routing | TopK on CPU | CUB Radix Sort |

## Preparation Checklist
- [ ] Install CUDA Toolkit (when hardware is ready).
- [ ] Add `.cu` files to `CMakeLists.txt` conditionally.
- [ ] Implement `Tensor::to(Device)` method for data transfer.
