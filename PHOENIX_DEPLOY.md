# Phoenix Deployment (Windows Server / WSL)

## ⚠️ NO SSH POLICY
**We do not use SSH for automated deployment.**
The workflow is strictly **Git-Based**.

---

## Deployment Workflow

### 1. Local Machine (Development)
Push your changes to GitHub.

```bash
git add .
git commit -m "Your update message"
git push origin main
```

### 2. Phoenix Host (Remote)
Login to Phoenix (WSL Terminal) manually and pull the changes.

```bash
cd ~/mm-rec-training
git pull origin main

# Update dependencies if needed
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_jax.txt

# Run Training
python mm_rec_jax/training/train_server_jax.py --config configs/moe_activation.json
```

---

## Setup (One Time)

### 1. Clone Repo on Phoenix
```bash
git clone https://github.com/obarlik/mm-rec.git ~/mm-rec-training
cd ~/mm-rec-training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify GPU
```bash
python -c "import jax; print(jax.devices())"
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
