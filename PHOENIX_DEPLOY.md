# Phoenix Deployment (Windows Server)

## Phoenix is Windows Server

Phoenix runs **Windows with cmd.exe**, not Linux bash.

---

## Quick Deploy (from Linux)

```bash
./deploy_to_phoenix.sh
```

This script handles Windows paths and commands automatically.

---

## Manual Deployment

### 1. Copy Files
```bash
scp -r mm_rec server client configs scripts requirements.txt \
  onurbarlik@hotmail.com@phoenix:mm-rec-training/
```

### 2. SSH to Phoenix
```bash
ssh onurbarlik@hotmail.com@phoenix
```

### 3. Setup (on Phoenix - Windows commands)
```cmd
cd mm-rec-training

REM Create virtual environment
python -m venv .venv

REM Activate
.venv\Scripts\activate

REM Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install dependencies
pip install -r requirements.txt
pip install -r server\requirements.txt
```

### 4. Verify GPU
```cmd
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

### 5. Start Server
```cmd
REM Foreground (testing)
python server\train_server.py

REM Background (production)
start /B python server\train_server.py > server.log 2>&1
```

---

## Usage from Linux

### Test Connection
```bash
python client/train_client.py --server http://phoenix:8000 --action health
```

### Submit Job
```bash
python client/train_client.py \
  --server http://phoenix:8000 \
  --action submit \
  --config configs/stage1_gpu.json
```

### Monitor
```bash
python client/train_client.py \
  --server http://phoenix:8000 \
  --action monitor \
  --job-id <job-id>
```

---

## Server Management

### Check if Running
```bash
ssh onurbarlik@hotmail.com@phoenix "tasklist | findstr python"
```

### View Logs
```bash
ssh onurbarlik@hotmail.com@phoenix "type mm-rec-training\server.log"
```

### Stop Server
```bash
ssh onurbarlik@hotmail.com@phoenix "taskkill /F /IM python.exe"
```

---

## Troubleshooting

### Python Not Found
```cmd
REM On Phoenix, check Python
where python
python --version
```

### CUDA Not Available
```cmd
REM Check NVIDIA driver
nvidia-smi

REM Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Port 8000 in Use
```cmd
REM Find process
netstat -ano | findstr :8000

REM Kill process (replace PID)
taskkill /F /PID <pid>
```

---

## Complete Workflow

```bash
# 1. Deploy
./deploy_to_phoenix.sh

# 2. Test
python client/train_client.py --server http://phoenix:8000 --action health

# 3. Submit
python client/train_client.py \
  --server http://phoenix:8000 \
  --action submit \
  --config configs/stage1_gpu.json

# 4. Monitor
python client/train_client.py \
  --server http://phoenix:8000 \
  --action monitor \
  --job-id abc12345

# 5. Download
python client/train_client.py \
  --server http://phoenix:8000 \
  --action download \
  --job-id abc12345 \
  --output models/stage1.pt
```

**Ready!** 🚀
