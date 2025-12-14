#!/usr/bin/env python3
"""
Foundation Dataset Preparation (Fixed Version v2)
Handles multiple HuggingFace cache locations
"""

import json
import random
import os
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("Foundation Dataset Preparation (Chat + Tool Use)")
print("=" * 80)

# Create directories - training server looks in workspace/data/
Path("workspace/data/raw").mkdir(parents=True, exist_ok=True)
Path("workspace/data/converted").mkdir(parents=True, exist_ok=True)

# ===== STEP 1: Find ShareGPT cache =====
print("\n📥 [1/5] Searching for ShareGPT cache...")

# Possible cache locations
cache_locations = [
    Path.home() / ".cache" / "huggingface" / "hub" / "datasets--RyokoAI--ShareGPT52K",
    Path("/mnt/d/AI-Models/huggingface/hub/datasets--RyokoAI--ShareGPT52K"),
    Path("/mnt/c/Users") / os.environ.get('USER', 'Onur') / ".cache" / "huggingface" / "hub" / "datasets--RyokoAI--ShareGPT52K",
]

# Add HF_HOME if set
if 'HF_HOME' in os.environ:
    hf_home = Path(os.environ['HF_HOME'])
    cache_locations.insert(0, hf_home / "hub" / "datasets--RyokoAI--ShareGPT52K")

snapshot_dir = None
for cache_loc in cache_locations:
    if cache_loc.exists():
        snapshots = list(cache_loc.glob("snapshots/*"))
        if snapshots:
            snapshot_dir = snapshots[0]
            print(f"✅ Found cache at: {cache_loc}")
            break

conversations = []

if snapshot_dir:
    # Parse from cache
    json_files = [f for f in snapshot_dir.glob("*.json") if f.name != "README.md"]
    
    for json_file in json_files:
        print(f"   Parsing {json_file.name}...")
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both single object and array formats
            if isinstance(data, list):
                conversations.extend(data)
            else:
                conversations.append(data)
                
        except Exception as e:
            print(f"   ⚠️ Skipped {json_file.name}: {e}")
    
    print(f"✅ Parsed {len(conversations)} conversations from cache")
else:
    print("⚠️ Cache not found. Using ToolBench only for this run.")
    print("   (You can download ShareGPT separately and re-run)")

# ===== STEP 2: Download ToolBench =====
print("\n📥 [2/5] Downloading ToolBench...")
try:
    from datasets import load_dataset
    toolbench = load_dataset('tuandunghcmut/toolbench-v1', split='train')
    print(f"✅ Downloaded {len(toolbench)} tool examples")
except Exception as e:
    print(f"❌ ToolBench download failed: {e}")
    print("   Continuing without tool examples...")
    toolbench = []

# ===== STEP 3: Convert ShareGPT =====
print("\n🔄 [3/5] Converting ShareGPT to unified format...")
converted_chat = []

for item in tqdm(conversations, desc="Processing ShareGPT"):
    if 'conversations' in item and isinstance(item['conversations'], list):
        messages = []
        for conv in item['conversations']:
            if isinstance(conv, dict):
                role = "user" if conv.get('from') in ['human', 'user'] else "assistant"
                content = conv.get('value') or conv.get('content', '')
                
                if content:  # Only add non-empty messages
                    messages.append({"role": role, "content": str(content)})
        
        # Only keep conversations with at least 2 turns
        if len(messages) >= 2:
            converted_chat.append({"messages": messages})

print(f"✅ Converted {len(converted_chat)} chat conversations")

# ===== STEP 4: Convert ToolBench =====
print("\n🔄 [4/5] Converting ToolBench to unified format...")
converted_tools = []

for item in tqdm(toolbench, desc="Processing ToolBench"):
    try:
        if 'query' in item or 'question' in item:
            user_query = item.get('query') or item.get('question', '')
            
            tool_response = (item.get('api_call') or 
                            item.get('tool_call') or 
                            item.get('function_call') or 
                            item.get('response') or
                            item.get('answer', ''))
            
            if user_query and tool_response:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant with tool access."},
                    {"role": "user", "content": str(user_query)},
                    {"role": "assistant", "content": str(tool_response)}
                ]
                converted_tools.append({"messages": messages})
    except Exception:
        continue

print(f"✅ Converted {len(converted_tools)} tool examples")

# ===== STEP 5: Combine and Shuffle =====
print("\n🎲 [5/5] Combining and shuffling datasets...")
combined = converted_chat + converted_tools

if not combined:
    print("❌ No data available! Both datasets failed.")
    print("   Please check HuggingFace connectivity or cache locations.")
    exit(1)

random.shuffle(combined)

# Save combined dataset - training server looks in workspace/data/
output_file = "workspace/data/combined_foundation.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Stats
stats = {
    "total_samples": len(combined),
    "chat_samples": len(converted_chat),
    "tool_samples": len(converted_tools),
    "chat_percentage": len(converted_chat) / len(combined) * 100 if combined else 0,
    "tool_percentage": len(converted_tools) / len(combined) * 100 if combined else 0
}

print("\n" + "=" * 80)
print("✅ Dataset Preparation Complete!")
print("=" * 80)
print(f"Total Samples: {stats['total_samples']:,}")
print(f"  - Chat: {stats['chat_samples']:,} ({stats['chat_percentage']:.1f}%)")
print(f"  - Tool: {stats['tool_samples']:,} ({stats['tool_percentage']:.1f}%)")
print(f"\nSaved to: {output_file}")
print("\nNext: Submit training with configs/foundation_chat_tools.json")
