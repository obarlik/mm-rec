#!/usr/bin/env python3
"""
Foundation Dataset Preparation (Fixed Version)
Manually parses ShareGPT JSON + downloads ToolBench
Handles Arrow conversion issues by direct JSON parsing
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("Foundation Dataset Preparation (Chat + Tool Use) - Fixed")
print("=" * 80)

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/converted").mkdir(parents=True, exist_ok=True)

# ===== STEP 1: Parse ShareGPT from cached JSON =====
print("\n📥 [1/5] Parsing ShareGPT from HuggingFace cache...")

# HuggingFace cache location (from error message)
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--RyokoAI--ShareGPT52K"

# Find snapshot directory
snapshots = list(cache_dir.glob("snapshots/*")) if cache_dir.exists() else []

if not snapshots:
    print("❌ Cache not found. Trying alternative dataset...")
    # Fallback: Download from alternative source
    print("   Downloading from alternative repository...")
    try:
        from datasets import load_dataset
        sharegpt = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train')
        conversations = [item for item in sharegpt if 'conversations' in item]
        print(f"✅ Downloaded {len(conversations)} conversations (alternative source)")
    except Exception as e:
        print(f"❌ Alternative download also failed: {e}")
        print("\n⚠️  Continuing with ToolBench only (chat-only dataset)")
        conversations = []
else:
    # Parse from cache
    snapshot_dir = snapshots[0]
    json_files = list(snapshot_dir.glob("*.json"))
    
    conversations = []
    for json_file in json_files:
        if json_file.name == "README.md":
            continue
        
        print(f"   Parsing {json_file.name}...")
        try:
            with open(json_file) as f:
                data = json.load(f)
                
            # Handle both single object and array formats
            if isinstance(data, list):
                conversations.extend(data)
            else:
                conversations.append(data)
                
        except Exception as e:
            print(f"   ⚠️ Skipped {json_file.name}: {e}")
    
    print(f"✅ Parsed {len(conversations)} conversations from cache")

# ===== STEP 2: Download ToolBench =====
print("\n📥 [2/5] Downloading ToolBench...")
try:
    from datasets import load_dataset
    toolbench = load_dataset('tuandunghcmut/toolbench-v1', split='train')
    print(f"✅ Downloaded {len(toolbench)} tool examples")
except Exception as e:
    print(f"❌ ToolBench download failed: {e}")
    toolbench = []

# ===== STEP 3: Convert ShareGPT =====
print("\n🔄 [3/5] Converting ShareGPT to unified format...")
converted_chat = []

for item in tqdm(conversations, desc="Processing"):
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

for item in tqdm(toolbench, desc="Processing"):
    # ToolBench format varies, adapt based on actual structure
    try:
        if 'query' in item or 'question' in item:
            user_query = item.get('query') or item.get('question', '')
            
            # Get tool response (try different field names)
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
        continue  # Skip malformed items

print(f"✅ Converted {len(converted_tools)} tool examples")

# ===== STEP 5: Combine and Shuffle =====
print("\n🎲 [5/5] Combining and shuffling datasets...")
combined = converted_chat + converted_tools
random.shuffle(combined)

if not combined:
    print("❌ No data to save! Both datasets failed.")
    exit(1)

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
print("\nNext step: Submit training job with configs/foundation_chat_tools.json")
