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
cxx_args = ['-O3', '-std=c++17', '-fopenmp']
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

# C++ only extensions (CPU fallback)
cpp_extensions = [
    CppExtension(
        'mm_rec_cpp_cpu',
        sources=[
            'src/mm_rec_block_cpp.cpp',
        ],
        extra_compile_args=cxx_args,
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

