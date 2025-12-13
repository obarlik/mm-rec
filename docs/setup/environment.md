# Environment Setup Guide

## Target Environment: Phoenix (WSL)

The primary training environment is the **Phoenix** server running **WSL (Windows Subsystem for Linux)**.

**Specs:**
- **Host**: Phoenix
- **User**: `onurbarlik@hotmail.com`
- **Path**: `~/mm-rec-training`
- **GPU**: NVIDIA RTX (Requires CUDA 12+)

## Prerequisites

### 1. Miniconda
We use Conda for environment management to isolate dependencies.
Ensure Conda is installed and initialized.

```bash
# Verify Conda
conda --version

# Activate Base (or specific env)
conda activate base
```

### 2. Git Configuration
The code is synchronized via GitHub. Ensure SSH keys are set up on Phoenix to pull from the repo.

```bash
# Check SSH Key
cat ~/.ssh/id_rsa.pub

# Test GitHub Connection
ssh -T git@github.com
```

## Installation Steps

### 1. Clone/Update Repository
```bash
cd ~
# If first time:
# git clone git@github.com:obarlik/mm-rec.git mm-rec-training

cd mm-rec-training
git pull origin main
```

### 2. Python Dependencies
We have separate requirement files for different backends.

**For JAX (Current/Recommended):**
```bash
pip install -r requirements_jax.txt
```
*Includes: `jax[cuda12]`, `flax`, `optax`.*

**For PyTorch (Legacy):**
```bash
pip install -r requirements.txt
```
*Includes: `torch`, `transformers`.*

### 3. Verify GPU Access
WSL requires specific NVIDIA drivers on Windows, which are passed through to Linux.

```bash
# Check NVIDIA System Management Interface
nvidia-smi

# Check JAX Element
python -c "import jax; print(jax.devices())"
```
