#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Nano Model Cycle${NC}"

# 1. Generate Data
echo -e "\n${GREEN}[1/3] Generating Pattern Data...${NC}"
python3 create_pattern_data.py

# 2. Train
echo -e "\n${GREEN}[2/3] Training Nano Model...${NC}"
# Clear old checkpoints to verify training from scratch
rm -f checkpoint_latest.bin checkpoint_best.bin
# Use the built executable
./build/mm_rec train config_nano.txt nano_data.bin

# 3. Infer (using the newly created checkpoint)
echo -e "\n${GREEN}[3/3] Testing Inference...${NC}"
# cmd_infer usage: <config> <model_path> <vocab> [prompt]
# Tokenizer maps "0" -> ID 4 (since 0-3 are special).
# Data uses 4..11. So promptecho "[3/3] Testing Inference (BPE Mode)..."
# Pass vocab.json. cmd_infer will look for merges.txt in the same dir.
./build/mm_rec infer config_nano.txt checkpoint_best.bin vocab.json "0 1 2" 

echo -e "\n${GREEN}✅ Cycle Complete!${NC}"
