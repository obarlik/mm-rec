#!/bin/bash
# Manual Phoenix Deployment Guide
# Run these commands step by step

echo "================================================"
echo "Manual Phoenix Deployment"
echo "================================================"

# Step 1: Create directory on Phoenix
echo ""
echo "Step 1: Creating directory on Phoenix..."
echo "Command: ssh onurbarlik@hotmail.com@phoenix 'mkdir -p ~/mm-rec-training'"
ssh onurbarlik@hotmail.com@phoenix 'mkdir -p ~/mm-rec-training'

# Step 2: Copy code (this will take a while)
echo ""
echo "Step 2: Copying code to Phoenix..."
echo "This may take 5-10 minutes..."
cd /home/onur/workspace/mm-rec
tar czf /tmp/mm-rec-code.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='checkpoints' \
  --exclude='*.so' \
  mm_rec/ server/ client/ configs/ scripts/ requirements.txt

scp /tmp/mm-rec-code.tar.gz onurbarlik@hotmail.com@phoenix:~/mm-rec-training/

# Step 3: Extract on Phoenix
echo ""
echo "Step 3: Extracting code..."
ssh onurbarlik@hotmail.com@phoenix 'cd ~/mm-rec-training && tar xzf mm-rec-code.tar.gz'

# Step 4: Setup Python
echo ""
echo "Step 4: Setting up Python environment..."
echo "This will take 10-15 minutes (PyTorch download)..."
ssh onurbarlik@hotmail.com@phoenix << 'EOF'
cd ~/mm-rec-training
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r server/requirements.txt
echo "✅ Python environment ready"
EOF

# Step 5: Verify GPU
echo ""
echo "Step 5: Verifying GPU..."
ssh onurbarlik@hotmail.com@phoenix << 'EOF'
cd ~/mm-rec-training
source .venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
EOF

# Step 6: Start server
echo ""
echo "Step 6: Starting training server..."
ssh onurbarlik@hotmail.com@phoenix << 'EOF'
cd ~/mm-rec-training
source .venv/bin/activate
pkill -f train_server.py || true
nohup python server/train_server.py > server.log 2>&1 &
echo "✅ Server started"
echo "Check logs: tail -f ~/mm-rec-training/server.log"
EOF

echo ""
echo "================================================"
echo "✅ Deployment Complete!"
echo "================================================"
echo ""
echo "Test from local machine:"
echo "  python client/train_client.py --server http://phoenix:8000 --action health"
echo ""
