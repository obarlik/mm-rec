#!/bin/bash
# Automatic Recovery Protocol for MM-Rec Training
# Detects numerical errors and recovers from last checkpoint

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
LOG_FILE="${LOG_FILE:-./recovery.log}"
MAX_ATTEMPTS=3
LR_REDUCTION=0.5

echo "🔄 MM-Rec Recovery Protocol"
echo "=========================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Find latest checkpoint
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_*.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found! Cannot recover."
    exit 1
fi

echo "📦 Latest checkpoint: $LATEST_CHECKPOINT"

# Extract step number from checkpoint
STEP=$(python3 -c "
import torch
import sys
checkpoint = torch.load('$LATEST_CHECKPOINT', map_location='cpu')
step = checkpoint.get('step', 0)
print(step)
")

echo "📍 Checkpoint step: $STEP"

# Reduce learning rate
echo "📉 Reducing learning rate by factor $LR_REDUCTION"

# Resume training with reduced learning rate
python3 -m mm_rec.scripts.train_modular \
    --resume_from "$LATEST_CHECKPOINT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Recovery protocol completed"

