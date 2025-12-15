# MM-Rec High-Performance C++ Implementation

Production-grade MM-Rec (Multi-Memory Recurrent Language Model) in C++ using LibTorch.

## Goals

- **3-5x faster** than Python/JAX baseline (1.4 it/s → 4-7 it/s)
- **<50ms inference** latency (vs 200ms Python)
- **Session persistence** - save/load memory state
- **Correctness** - match Python outputs (max_diff < 1e-4)

## Architecture

- **GRU-Style Gating**: Update/reset/candidate gates per layer
- **UBOO Deep Supervision**: Every layer predicts next token
- **Per-Layer Memory**: Isolated state for hierarchical abstraction
- **LibTorch Backend**: Direct PyTorch C++ API

## Build

```bash
# Prerequisites
sudo apt install cmake build-essential intel-mkl libopenblas-dev

# Download LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip -d /opt/

# Build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=ON
make -j$(nproc)

# Test
./tests/test_gated_memory
./tests/test_per_layer_state
```

## Performance Targets

| Component | Python/JAX | C++ Target | Status |
|-----------|------------|------------|--------|
| Training Speed | 1.4 it/s | 4-7 it/s | TBD |
| Inference Latency | ~200ms | <50ms | TBD |
| Associative Scan | baseline | 10-27x faster | Proven in Python bindings |
| Memory Usage | baseline | <10% overhead | TBD |

## Project Status

Phase 1: MVP Implementation (Week 1-2) - **IN PROGRESS**

- [x] Project structure created
- [ ] GRU-style gated memory
- [ ] UBOO output projections
- [ ] Per-layer state management
- [ ] Basic forward pass
- [ ] Unit tests

Based on 8+ months production Python/JAX experience with 27 training epochs.
