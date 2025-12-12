#!/bin/bash
# Deploy to Phoenix WSL (direct connection, no password)

set -e

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="phoenix"
REMOTE_DIR="~/mm-rec-training"
LOCAL_DIR="/home/onur/workspace/mm-rec"

echo "================================================"
echo "Deploy to Phoenix WSL"
echo "================================================"

# Note: SSH may ask for password (key auth not fully working)
# This is OK, just enter password once per deployment

# 1. Create directory (via WSL)
echo ""
echo "[1/5] Creating directory..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "wsl bash -c 'mkdir -p ${REMOTE_DIR}'"

# 2. Sync code
echo ""
echo "[2/5] Syncing code..."
rsync -az \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'checkpoints' \
  --exclude '*.so' \
  ${LOCAL_DIR}/ \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
echo "✅ Code synced"

# 3. Setup Python (via WSL)
echo ""
echo "[3/5] Setting up Python..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "wsl bash -c 'cd ${REMOTE_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install -r server/requirements.txt'"

# 4. Verify GPU (via WSL)
echo ""
echo "[4/5] Checking GPU..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "wsl bash -c 'cd ${REMOTE_DIR} && source .venv/bin/activate && python -c \"import torch; print(f'\"'\"'CUDA: {torch.cuda.is_available()}'\"'\"'); print(f'\"'\"'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}'\"'\"')\"'"

# 5. Start server (via WSL)
echo ""
echo "[5/5] Starting server..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "wsl bash -c 'cd ${REMOTE_DIR} && source .venv/bin/activate && pkill -f train_server.py || true && nohup python server/train_server.py > server.log 2>&1 &'"

echo ""
echo "================================================"
echo "✅ Deployment Complete!"
echo "================================================"
echo ""
echo "📡 Server: http://phoenix:8000"
echo ""
echo "🔧 Test:"
echo "  python client/train_client.py --server http://phoenix:8000 --action health"
echo ""
