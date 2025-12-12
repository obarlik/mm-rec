# Windows 11 Setup Guide

## Quick Start

### 1. Install Python 3.12
```powershell
# Download from python.org or use winget
winget install Python.Python.3.12
```

### 2. Install CUDA Toolkit
```powershell
# Download from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
# Choose: Windows > x86_64 > 11 > exe (local)
```

### 3. Clone Repository
```powershell
git clone <your-repo-url> mm-rec
cd mm-rec
```

### 4. Setup Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 5. Install PyTorch with CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 7. Verify GPU
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Training Commands

### Download Data (if needed)
```powershell
python scripts\download_phase1_data.py --output-dir data\phase1
```

### Run Stage 1 GPU Training
```powershell
python scripts\train_stage1_gpu.py
```

### Monitor Progress
Training will show updates every 10 seconds automatically.

## Troubleshooting

### CUDA Not Found
```powershell
# Check CUDA installation
nvcc --version

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
```powershell
# Reduce batch size in script
# Edit train_stage1_gpu.py: BATCH_SIZE = 8  # or 4
```

### Slow Performance
```powershell
# Check GPU usage
nvidia-smi

# Enable mixed precision (should be default)
# USE_MIXED_PRECISION = True in script
```

## File Paths (Windows)

- Scripts: `scripts\`
- Data: `data\phase1\`
- Checkpoints: `checkpoints\progressive\`
- Logs: `logs\`

## Next Steps

After Stage 1 completes:
```powershell
python scripts\train_stage2.py --stage1-checkpoint checkpoints\progressive\stage1_gpu_final.pt
```
