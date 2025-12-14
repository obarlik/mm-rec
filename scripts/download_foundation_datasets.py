#!/usr/bin/env python3
"""
Download Foundation Training Datasets
Downloads ShareGPT-90K and ToolBench from HuggingFace
"""

import os
import sys
from pathlib import Path

# Ensure datasets library is available
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# Create data directory
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Foundation Dataset Download")
print("=" * 80)

# 1. ShareGPT-90K (Chat)
print("\n📥 Downloading ShareGPT-90K...")
try:
    sharegpt = load_dataset('RyokoAI/ShareGPT52K', split='train')
    output_file = data_dir / "sharegpt_90k.jsonl"
    sharegpt.to_json(str(output_file))
    print(f"✅ ShareGPT saved: {output_file}")
    print(f"   Size: {len(sharegpt)} conversations")
except Exception as e:
    print(f"❌ ShareGPT download failed: {e}")
    sys.exit(1)

# 2. ToolBench (Function Calling)
print("\n📥 Downloading ToolBench...")
try:
    toolbench = load_dataset('tuandunghcmut/toolbench-v1', split='train')
    output_file = data_dir / "toolbench.jsonl"
    toolbench.to_json(str(output_file))
    print(f"✅ ToolBench saved: {output_file}")
    print(f"   Size: {len(toolbench)} tool examples")
except Exception as e:
    print(f"❌ ToolBench download failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ All datasets downloaded successfully!")
print("=" * 80)
print("\nNext step: Run format conversion script")
