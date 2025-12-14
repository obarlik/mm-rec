#!/bin/bash
# Foundation Dataset Setup - Remote Execution Script
# Run this on Phoenix via RDP/SSH terminal

set -e  # Exit on error

echo "=================================="
echo "Foundation Dataset Preparation"
echo "=================================="

# Navigate to project
cd ~/mm-rec-training

# Pull latest code
echo "[1/5] Pulling latest code from GitHub..."
git pull origin main

# Activate conda environment
echo "[2/5] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mm-rec

# Install required packages
echo "[3/5] Installing required packages..."
pip install -q datasets huggingface_hub tqdm

# Run dataset preparation
echo "[4/5] Downloading and preparing datasets..."
echo "This will take 30-60 minutes depending on internet speed..."
python scripts/prepare_foundation_data.py

# Check if dataset was created successfully
if [ ! -f "data/combined_foundation.jsonl" ]; then
    echo "❌ Error: Dataset preparation failed!"
    exit 1
fi

echo "[5/5] Dataset ready! Checking size..."
wc -l data/combined_foundation.jsonl

echo ""
echo "✅ Dataset preparation complete!"
echo ""
echo "Next step: Submit training job"
echo "Run: python client/train_client.py --server http://localhost:8090 --action submit --config configs/foundation_chat_tools.json"
