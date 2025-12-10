"""
Setup script for MM-Rec C++ extensions.

Build command:
    python setup.py build_ext --inplace

Or install:
    pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

# Check if CUDA is available
try:
    import torch
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False

# Base compile arguments
# Get PyTorch library path for rpath and library paths
import torch
import os
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
rpath_flag = f'-Wl,-rpath,{torch_lib}'
library_dirs = [torch_lib] if os.path.exists(torch_lib) else []

# Modern CPU optimizations - MAXIMUM PERFORMANCE
cxx_args = [
    '-O3',                    # Maximum optimization
    '-std=c++17',             # C++17 standard
    '-march=native',          # Use native CPU architecture (auto-detect AVX-512, AVX2, etc.)
    '-mtune=native',          # Tune for native CPU
    '-fopenmp',               # OpenMP support for parallel processing
    '-mavx',                  # AVX instructions
    '-mavx2',                 # AVX2 instructions (8 floats)
    # AVX-512 flags - only if CPU supports (will fail gracefully if not)
    # '-mavx512f',              # AVX-512 foundation (16 floats) - optional
    # '-mavx512cd',             # AVX-512 conflict detection - optional
    '-mfma',                  # FMA (Fused Multiply-Add) instructions
    '-msse4.2',               # SSE4.2 instructions
    '-funroll-loops',         # Loop unrolling
    '-ffast-math',            # Fast math (with care for numerical stability)
    '-fno-math-errno',        # Don't set errno for math functions
    '-flto',                  # Link-time optimization
    '-fno-strict-aliasing',   # Allow type punning (careful!)
    rpath_flag
]
nvcc_args = ['-O3', '--use_fast_math']

# Add architecture-specific flags
if cuda_available:
    # Detect CUDA architecture
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Parse compute capability and add to nvcc args
            compute_cap = result.stdout.strip().split('\n')[0]
            if compute_cap:
                nvcc_args.append(f'-arch=sm_{compute_cap.replace(".", "")}')
    except:
        pass
    
    # Default to sm_75 if detection fails
    if not any('arch=sm_' in arg for arg in nvcc_args):
        nvcc_args.append('-arch=sm_75')

# Define extensions
extensions = []

# C++ only extensions (CPU fallback with modern optimizations)
cpp_extensions = [
    CppExtension(
        'mm_rec_cpp_cpu',
        sources=[
            'src/mm_rec_block_cpp.cpp',
        ],
        extra_compile_args=cxx_args,
        extra_link_args=['-fopenmp', rpath_flag] if 'rpath_flag' in locals() else ['-fopenmp'],
        library_dirs=library_dirs if 'library_dirs' in locals() else [],
        language='c++'
    ),
    CppExtension(
        'mm_rec_scan_cpu',
        sources=[
            'src/associative_scan_cpu.cpp',
            'src/core/exp_log_simd.cpp',
            'src/core/blelloch_scan_parallel.cpp',
            'src/core/fast_math_asm.cpp',  # Assembly optimizations
            'src/core/log_exp_conversion_simd.cpp',  # SIMD log/exp conversion
        ],
        extra_compile_args=cxx_args,
        extra_link_args=['-fopenmp', rpath_flag] if 'rpath_flag' in locals() else ['-fopenmp'],
        library_dirs=library_dirs if 'library_dirs' in locals() else [],
        include_dirs=['src'],  # For header includes
        language='c++'
    ),
    CppExtension(
        'mm_rec_blocks_cpu',
        sources=[
            'src/core/exp_log_simd.cpp',  # SIMD functions needed
            'src/blocks/core_recurrence_fused.cpp',
            'src/blocks/mdi_cpu_optimized.cpp',
            'src/bindings/python_bindings.cpp',
            'src/core/blas_wrapper.cpp',  # BLAS wrapper (MKL/OpenBLAS or manual)
        ],
        extra_compile_args=cxx_args,
        extra_link_args=['-fopenmp', rpath_flag] if 'rpath_flag' in locals() else ['-fopenmp'],
        library_dirs=library_dirs if 'library_dirs' in locals() else [],
        include_dirs=['src'],  # For header includes
        language='c++'
    )
]

# CUDA extensions (if available)
if cuda_available:
    cuda_extensions = [
        CUDAExtension(
            'mm_rec_cpp_cuda',
            sources=[
                'src/associative_scan_cuda.cpp',
                'cuda/associative_scan_kernel.cu',
                'cuda/mdi_kernel.cu',
                'cuda/memory_state_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        )
    ]
    extensions.extend(cuda_extensions)
else:
    print("⚠️  CUDA not available, building CPU-only extensions")

extensions.extend(cpp_extensions)

setup(
    name='mm_rec_cpp',
    version='0.1.0',
    description='MM-Rec C++ Optimizations',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.8',
)

