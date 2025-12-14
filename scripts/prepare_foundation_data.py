#!/usr/bin/env python3
"""
Foundation Dataset Preparation
Downloads ShareGPT + ToolBench, converts to unified format
Run on Phoenix with: python scripts/prepare_foundation_data.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

print("=" * 80)
print("Foundation Dataset Preparation (Chat + Tool Use)")
print("=" * 80)

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/converted").mkdir(parents=True, exist_ok=True)

# ===== STEP 1: Download ShareGPT-90K =====
print("\n📥 [1/5] Downloading ShareGPT-90K...")
try:
    sharegpt = load_dataset('RyokoAI/ShareGPT52K', split='train')
    print(f"✅ Downloaded {len(sharegpt)} conversations")
except Exception as e:
    print(f"❌ ShareGPT download failed: {e}")
    print("Tip: Check internet connection and HuggingFace access")
    exit(1)

# ===== STEP 2: Download ToolBench =====
print("\n📥 [2/5] Downloading ToolBench...")
try:
    toolbench = load_dataset('tuandunghcmut/toolbench-v1', split='train')
    print(f"✅ Downloaded {len(toolbench)} tool examples")
except Exception as e:
    print(f"❌ ToolBench download failed: {e}")
    exit(1)

# ===== STEP 3: Convert ShareGPT =====
print("\n🔄 [3/5] Converting ShareGPT to unified format...")
converted_chat = []

for item in tqdm(sharegpt, desc="Processing"):
    if 'conversations' in item:
        messages = []
        for conv in item['conversations']:
            role = "user" if conv.get('from') == 'human' else "assistant"
            messages.append({"role": role, "content": conv.get('value', '')})
        
        # Only keep conversations with at least 2 turns
        if len(messages) >= 2:
            converted_chat.append({"messages": messages})

print(f"✅ Converted {len(converted_chat)} chat conversations")

# ===== STEP 4: Convert ToolBench =====
print("\n🔄 [4/5] Converting ToolBench to unified format...")
converted_tools = []

for item in tqdm(toolbench, desc="Processing"):
    # ToolBench format varies, adapt based on actual structure
    # Common fields: query, api_call, tool_call, function_call
    
    if 'query' in item:
        user_query = item['query']
        
        # Get tool response (try different field names)
        tool_response = (item.get('api_call') or 
                        item.get('tool_call') or 
                        item.get('function_call') or 
                        item.get('response', ''))
        
        if tool_response:
            messages = [
                {"role": "system", "content": "You are a helpful assistant with tool access."},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": str(tool_response)}
            ]
            converted_tools.append({"messages": messages})

print(f"✅ Converted {len(converted_tools)} tool examples")

# ===== STEP 5: Combine and Shuffle =====
print("\n🎲 [5/5] Combining and shuffling datasets...")
combined = converted_chat + converted_tools
random.shuffle(combined)

# Save combined dataset
output_file = "data/combined_foundation.jsonl"
with open(output_file, 'w') as f:
    for item in combined:
        f.write(json.dumps(item) + '\n')

# Save stats
stats = {
    "total_samples": len(combined),
    "chat_samples": len(converted_chat),
    "tool_samples": len(converted_tools),
    "chat_percentage": len(converted_chat) / len(combined) * 100,
    "tool_percentage": len(converted_tools) / len(combined) * 100
}

print("\n" + "=" * 80)
print("✅ Dataset Preparation Complete!")
print("=" * 80)
print(f"Total Samples: {stats['total_samples']:,}")
print(f"  - Chat: {stats['chat_samples']:,} ({stats['chat_percentage']:.1f}%)")
print(f"  - Tool: {stats['tool_samples']:,} ({stats['tool_percentage']:.1f}%)")
print(f"\nSaved to: {output_file}")
print("\nNext step: Submit training job with configs/foundation_chat_tools.json")
