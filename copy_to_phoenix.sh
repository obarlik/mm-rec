#!/bin/bash
# Quick code copy - only essential files

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="phoenix"

echo "� Creating small archive (only source code)..."

cd /home/onur/workspace/mm-rec

# Only essential directories
tar czf /tmp/mm-rec-small.tar.gz \
  mm_rec/ \
  server/ \
  client/ \
  configs/ \
  scripts/*.py \
  requirements.txt \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.so'

SIZE=$(du -h /tmp/mm-rec-small.tar.gz | cut -f1)
echo "✅ Archive created: $SIZE"

echo ""
echo "📤 Uploading to Phoenix..."
scp /tmp/mm-rec-small.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:~/mm-rec.tar.gz

echo ""
echo "✅ Done! Now on Phoenix:"
echo ""
echo "ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "tar xzf mm-rec.tar.gz"
echo "cd mm_rec"
echo "python3 -m venv .venv"
echo "source .venv/bin/activate"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo "pip install -r requirements.txt"
echo "cd ../server && python train_server.py"
echo ""
