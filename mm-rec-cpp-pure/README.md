# MM-Rec Pure C++ (No LibTorch)

Minimal dependency implementation of MM-Rec using only MKL/OpenBLAS.

## Features

✅ **Zero LibTorch Dependency**
- Custom Tensor class
- Direct MKL/BLAS calls
- SIMD-optimized activations

✅ **Small Binary Size**
- ~50-100MB (vs 500MB with LibTorch)
- Minimal runtime dependencies

✅ **Same Performance**
- MKL for matmul
- OpenMP parallelization  
- Production-tested formulas

## Build

```bash
cd mm-rec-cpp-pure
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Test
./test_tensor
./test_gated_memory
```

## Dependencies

**Required**:
- CMake 3.18+
- g++ 13+ (C++17)
- OpenMP
- Intel MKL or OpenBLAS

**Install**:
```bash
sudo apt install cmake build-essential intel-mkl libgomp1
```

## Status

- [x] Tensor class (MKL BLAS)
- [x] Linear layer
- [x] GatedMemoryUpdate (GRU-style)
- [x] Tests passing
- [ ] Full model
- [ ] Inference

## Binary Size Comparison

| Version | Size | Dependencies |
|---------|------|--------------|
| **Pure C++** | **~50-100MB** | MKL/OpenBLAS only |
| LibTorch | ~500MB | Full PyTorch |

## Performance

Same or better than LibTorch due to:
- Direct MKL calls (no overhead)
- SIMD optimizations
- Minimal abstraction layers
