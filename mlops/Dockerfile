# MM-Rec 100M Modular Training Dockerfile
# Includes Triton kernels and optimized dependencies

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
RUN pip3 install triton

# Install other dependencies
RUN pip3 install \
    numpy \
    tqdm \
    transformers \
    datasets \
    wandb \
    tensorboard \
    safetensors

# Install MM-Rec package
WORKDIR /workspace/mm-rec
COPY . /workspace/mm-rec/
RUN pip3 install -e .

# Set working directory
WORKDIR /workspace/mm-rec

# Default command
CMD ["python3", "-m", "mm_rec.scripts.train_modular", "--stage", "all"]

