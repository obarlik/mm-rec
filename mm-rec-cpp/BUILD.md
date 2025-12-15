# MM-Rec C++ Build Instructions

## Quick Start (Using Python's LibTorch)

```bash
cd mm-rec-cpp

# Create build directory
mkdir -p build && cd build

# Configure with Python's LibTorch
python3 -c "import torch; print(torch.utils.cmake_prefix_path)" | xargs -I {} cmake .. -DCMAKE_PREFIX_PATH={} -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
./tests/test_gated_memory
./tests/test_per_layer_state
```

## Installing Standalone LibTorch (Alternative)

If you want standalone LibTorch

:
```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip -d /opt/

# Configure with standalone
cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

## Troubleshooting

### LibTorch not found
- Check Python PyTorch: `python3 -c "import torch; print(torch.__version__)"`
- Get CMake path: `python3 -c "import torch; print(torch.utils.cmake_prefix_path)"`

### MKL not found
- Install: `sudo apt install intel-mkl` (or use OpenBLAS)
- CMake will auto-fallback to LibTorch's BLAS

### Build errors
- Ensure C++17: `g++ --version` (need 7+)
- Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"` (need 2.0+)
