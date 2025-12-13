
import os
import torch
import torch.nn as nn
import sys
from pathlib import Path
import json

# Setup paths
workspace_dir = Path("workspace").absolute()
sys.path.insert(0, str(workspace_dir))

# Enable debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
try:
    torch.autograd.set_detect_anomaly(True)
except:
    pass

def test_gpu():
    print("🔍 Starting GPU Diagnostic Test...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')

    # 1. Test Model Initialization
    print("\n1. Testing Model Initialization...")
    try:
        from mm_rec.model import MMRecModel
        from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.vocab_size + 1000
        print(f"   Vocab Size: {vocab_size}")
        
        model = MMRecModel(
            vocab_size=vocab_size,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            ffn_dim=128
        ).to(device)
        print("   ✅ Model initialized on GPU")
    except Exception as e:
        print(f"   ❌ Init failed: {e}")
        return

    # 2. Test Dummy Forward/Backward
    print("\n2. Testing Dummy Data (Random)...")
    try:
        input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
        labels = input_ids.clone()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        # MMRecModel returns logits only, handle loss manually
        logits = model(input_ids)
        
        # Compute loss (Shift labels like CausalLM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        print(f"   Forward Pass Loss: {loss.item():.4f}")
        
        loss.backward()
        print("   ✅ Backward Pass OK")
        
        optimizer.step()
        print("   ✅ Optimizer Step OK")
    except Exception as e:
        print(f"   ❌ Dummy test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test Real Data Tokenization
    print("\n3. Testing Real Data Tokenization...")
    try:
        data_path = workspace_dir / "data" / "phase1" / "train.json"
        
        if not data_path.exists():
             print(f"   ⚠️ Data file not found at {data_path}. Skipping.")
             return

        with open(data_path) as f:
            data = json.load(f)[:10] # Test first 10
            
        print(f"   Loaded {len(data)} examples")
        
        max_idx = 0
        for i, item in enumerate(data):
            text = ""
            for msg in item['conversations']:
                text += list(msg.values())[1] # content
            
            ids = tokenizer.encode(text)
            if ids:
                max_idx = max(max_idx, max(ids))
            
            if max_idx >= vocab_size:
                print(f"   ❌ FOUND BAD TOKEN: {max_idx} >= {vocab_size} in sample {i}")
                return
        
        print(f"   ✅ Max Token ID in data: {max_idx} (Safe < {vocab_size})")

    except Exception as e:
        print(f"   ❌ Data test failed: {e}")
        traceback.print_exc()
        return

    print("\n🎉 ALL TESTS PASSED! GPU IS READY.")

if __name__ == "__main__":
    test_gpu()
