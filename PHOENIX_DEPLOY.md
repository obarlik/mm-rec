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

### 2. Phoenix Host (Automated via Gateway)

**Option A: Hot Reload (Recommended)**
```bash
# From your local machine
python client/train_client.py --server http://phoenix:8090 --action update
```
Gateway automatically pulls code and restarts all servers.

**Option B: Manual Restart (if Gateway is down)**
```bash
# SSH/RDP to Phoenix, then:
cd ~/mm-rec-training
git pull origin main

# Kill old processes
pkill -9 -f python

# Restart Gateway (manages everything)
./phoenix_manual_start.sh
```

Gateway launches:
- Training Server (Port 8001, GPU)
- Inference Server (Port 8002, CPU)

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

## Server Management (via Gateway)

The Gateway (Port 8090) manages the Training Server (Port 8001) and Inference Server (Port 8002).

### 1. View System Status
```bash
python client/train_client.py --server http://phoenix:8090 --action health
```

### 2. View Logs
View the internal logs of the underlying servers.

```bash
# Gateway Log
python client/train_client.py --server http://phoenix:8090 --action logs --type gateway

# Training Server Log
python client/train_client.py --server http://phoenix:8090 --action logs --type server
```

### 3. Update Code (Hot Reload)
This triggers a `git pull` on the server and restarts the training process automatically.

```bash
python client/train_client.py --server http://phoenix:8090 --action update
```

### 4. Direct Inference (JAX)
Talk to the model directly via Gateway. (Requires `client/chat_client.py` - Coming Soon)

```bash
curl -X POST http://phoenix:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello world"}'
```

---

## Troubleshooting

### If Gateway is Down
If `curl http://phoenix:8090` fails, you must access the terminal manually:

```bash
# On Phoenix Terminal:
cd ~/mm-rec-training
./phoenix_manual_start.sh
```

### Force Restart
```bash
# On Phoenix Terminal:
taskkill /F /IM python.exe
./phoenix_manual_start.sh
```
