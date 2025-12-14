# Foundation Dataset Preparation - Manual Instructions (Phoenix)

## Background
Due to local Python environment restrictions, dataset download needs to be performed on Phoenix server where miniconda environment allows package installation.

---

## Step 1: Activate Phoenix Environment

**On Phoenix Terminal (WSL):**
```bash
cd ~/mm-rec-training
conda activate mm-rec
```

---

## Step 2: Install Datasets Library

```bash
pip install datasets huggingface_hub
```

---

## Step 3: Download ShareGPT-90K

**Option A: Python Script (Recommended)**
```bash
python -c "
from datasets import load_dataset
import json

print('Downloading ShareGPT-90K...')
ds = load_dataset('RyokoAI/ShareGPT52K', split='train')
print(f'Downloaded {len(ds)} conversations')

# Save as JSONL
with open('data/raw_sharegpt_90k.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps(item) + '\n')
        
print('✅ ShareGPT saved to data/raw_sharegpt_90k.jsonl')
"
```

**Option B: Command Line**
```bash
mkdir -p data
huggingface-cli download RyokoAI/ShareGPT52K --repo-type dataset --local-dir data/sharegpt_cache
```

---

## Step 4: Download ToolBench

```bash
python -c "
from datasets import load_dataset
import json

print('Downloading ToolBench...')
ds = load_dataset('tuandunghcmut/toolbench-v1', split='train')
print(f'Downloaded {len(ds)} tool examples')

# Save as JSONL
with open('data/raw_toolbench.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps(item) + '\n')
        
print('✅ ToolBench saved to data/raw_toolbench.jsonl')
"
```

---

## Step 5: Verify Downloads

```bash
wc -l data/raw_*.jsonl
head -1 data/raw_sharegpt_90k.jsonl | python -m json.tool
head -1 data/raw_toolbench.jsonl | python -m json.tool
```

Expected output:
- `raw_sharegpt_90k.jsonl`: ~90,000 lines
- `raw_toolbench.jsonl`: ~16,000 lines

---

## Step 6: Format Conversion

**Create conversion script on Phoenix:**
```bash
cat > scripts/convert_foundation_data.py << 'EOF'
#!/usr/bin/env python3
"""
Convert ShareGPT + ToolBench to unified format
"""
import json
from pathlib import Path

def convert_sharegpt(input_file, output_file):
    """Convert ShareGPT format to our chat format"""
    print(f"Converting ShareGPT from {input_file}...")
    count = 0
    
    with open(input_file) as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            
            # ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
            # Our format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            
            if 'conversations' in data:
                messages = []
                for conv in data['conversations']:
                    role = "user" if conv.get('from') == 'human' else "assistant"
                    messages.append({"role": role, "content": conv.get('value', '')})
                
                # Only keep conversations with at least 2 turns
                if len(messages) >= 2:
                    fout.write(json.dumps({"messages": messages}) + '\n')
                    count += 1
                    
    print(f"✅ Converted {count} conversations")
    return count

def convert_toolbench(input_file, output_file):
    """Convert ToolBench format to our format"""
    print(f"Converting ToolBench from {input_file}...")
    count = 0
    
    with open(input_file) as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            
            # ToolBench has various formats, adapt as needed
            # Basic structure: query + tool_call
            if 'query' in data and ('api_call' in data or 'tool_call' in data):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant with tool access."},
                    {"role": "user", "content": data['query']},
                    {"role": "assistant", "content": data.get('api_call') or data.get('tool_call', '')}
                ]
                
                fout.write(json.dumps({"messages": messages}) + '\n')
                count += 1
                
    print(f"✅ Converted {count} tool examples")
    return count

# Main conversion
sharegpt_count = convert_sharegpt('data/raw_sharegpt_90k.jsonl', 'data/conv_sharegpt.jsonl')
toolbench_count = convert_toolbench('data/raw_toolbench.jsonl', 'data/conv_toolbench.jsonl')

# Combine and shuffle
print("\nCombining datasets...")
import random
combined = []

with open('data/conv_sharegpt.jsonl') as f:
    combined.extend([line for line in f])
    
with open('data/conv_toolbench.jsonl') as f:
    combined.extend([line for line in f])

random.shuffle(combined)

# Save combined
with open('data/combined_foundation.jsonl', 'w') as f:
    f.writelines(combined)

print(f"\n✅ Final dataset: {len(combined)} examples")
print(f"   - ShareGPT: {sharegpt_count} ({sharegpt_count/len(combined)*100:.1f}%)")
print(f"   - ToolBench: {toolbench_count} ({toolbench_count/len(combined)*100:.1f}%)")
print(f"\nSaved to: data/combined_foundation.jsonl")
EOF

chmod +x scripts/convert_foundation_data.py
python scripts/convert_foundation_data.py
```

---

## Step 7: Create Training Config

```bash
cat > configs/foundation_chat_tools.json << 'EOF'
{
    "job_name": "foundation_chat_tools",
    "model_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "ffn_dim": 2048,
    "num_epochs": 40,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "max_length": 512,
    "vocab_size": 100300
}
EOF
```

---

## Step 8: Launch Training

```bash
# Stop current baseline training (optional)
# curl -X POST http://localhost:8090/api/train/stop/797cc451

# Test Gateway health
curl http://localhost:8090/gateway/health

# Submit new foundation training
python client/train_client.py \
    --server http://localhost:8090 \
    --action submit \
    --config configs/foundation_chat_tools.json
```

---

## Troubleshooting

**If download fails:**
```bash
# Check HuggingFace connectivity
ping huggingface.co

# Try with explicit cache dir
export HF_HOME=~/mm-rec-training/.cache/huggingface
```

**If conversion fails:**
```bash
# Inspect raw data formats
head -3 data/raw_sharegpt_90k.jsonl | python -m json.tool
head -3 data/raw_toolbench.jsonl | python -m json.tool

# Adjust convert_foundation_data.py accordingly
```

---

## Next Steps After Training Launch

1. Monitor first epoch logs
2. Verify loss convergence (~9.0 → ~1.5)
3. Test with chat_client.py every 10 epochs
4. Evaluate both chat and tool use capabilities
