#!/bin/bash
# Run JAX Benchmark on Phoenix

set -e

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="phoenix"
REMOTE_DIR="~/mm-rec-training"
LOCAL_DIR="/home/onur/workspace/mm-rec"

echo "================================================"
echo "Run JAX Benchmark on Phoenix"
echo "================================================"

# 1. Sync code (Fast)
echo ""
echo "[1/3] Syncing code..."
rsync -az \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'checkpoints' \
  ${LOCAL_DIR}/ \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

# 2. Install JAX deps (Ensure installed)
echo ""
echo "[2/3] Checking JAX dependencies..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && source .venv/bin/activate && pip install -r requirements_jax.txt"

# 3. Run Benchmark
echo ""
echo "[3/3] Running JAX Training Benchmark..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && source .venv/bin/activate && python mm_rec_jax/training/train_server_jax.py"
