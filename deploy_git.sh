#!/bin/bash
# Deploy to Phoenix using Git (FAST!)

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="192.168.1.86"  # Phoenix IP
REPO_URL="https://github.com/obarlik/mm-rec.git"

echo "🚀 Deploying to Phoenix via Git Clone"
echo ""

# Clone and setup on Phoenix
ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
# Remove existing directory if present
rm -rf mm-rec-training

# Clone repo
git clone ${REPO_URL} mm-rec-training
cd mm-rec-training

# Setup Python
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
pip install -r server/requirements.txt

# Build C++ extensions (if compiler available)
echo "Building C++ extensions..."
cd mm_rec/cpp
if python setup.py build_ext --inplace 2>/dev/null; then
    echo "✅ C++ extensions built successfully"
else
    echo "⚠️  C++ build failed, using Python fallback"
fi
cd ../..

# Verify GPU
python -c 'import torch; print(f"CUDA: {torch.cuda.is_available()}"); print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'

# Start server
nohup python server/train_server.py > server.log 2>&1 &

echo ""
echo "✅ Server started!"
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
