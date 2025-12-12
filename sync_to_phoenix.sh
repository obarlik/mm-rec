#!/bin/bash
# Fastest method: Use rsync with progress

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="phoenix"
REMOTE_DIR="mm-rec-training"

echo "🚀 Fast sync to Phoenix (rsync with progress)"
echo ""

# Sync directly with rsync (shows progress)
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='checkpoints' \
  --exclude='data' \
  --exclude='experiments' \
  --exclude='*.so' \
  --exclude='build' \
  /home/onur/workspace/mm-rec/ \
  ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_DIR}/

echo ""
echo "✅ Sync complete!"
echo ""
echo "Now on Phoenix:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  python3 -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo "  pip install -r requirements.txt -r server/requirements.txt"
echo "  python server/train_server.py"
echo ""
