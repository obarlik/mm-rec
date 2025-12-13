import time
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.data.chat_format import ChatMessage
from mm_rec.data.dataset import SFTDataset, SFTDataCollator
from mm_rec.training.sft_trainer import SFTConfig

def benchmark():
    WORKSPACE_DIR = Path(".").resolve()
    print(f"📂 Workspace: {WORKSPACE_DIR}")

    # Setup
    tokenizer = get_tokenizer()
    sft_config = SFTConfig(max_length=256)
    
    # Load data
    possible_paths = [
        WORKSPACE_DIR / "workspace" / "data" / "train.json", 
        WORKSPACE_DIR / "workspace" / "data" / "phase1" / "train.json",
        WORKSPACE_DIR / "data" / "train.json"
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
            
    if not data_path:
        print(f"❌ Data not found in {possible_paths}")
        return

    print(f"ℹ️  Loading data from: {data_path}")
    start_load = time.time()
    with open(data_path) as f:
        data = json.load(f)
    print(f"⏱️  JSON Load: {time.time() - start_load:.4f}s")

    conversations = []
    for item in data:
        messages = [ChatMessage(role=msg['role'], content=msg['content']) for msg in item['conversations']]
        conversations.append(messages)
    print(f"ℹ️  Loaded {len(conversations)} items")

    # Benchmark __getitem__
    dataset = SFTDataset(conversations, tokenizer, sft_config)
    print("🚀 Benchmarking __getitem__ (Single Thread)...")
    start_time = time.time()
    for i in range(100):
        _ = dataset[i]
    duration = time.time() - start_time
    print(f"⏱️  100 items: {duration:.4f}s")
    print(f"⚡ Speed: {100/duration:.2f} items/s") # Should be > 1000

    # Benchmark DataLoader
    print("🚀 Benchmarking DataLoader (num_workers=4)...")
    collator = SFTDataCollator(tokenizer, ignore_index=-100)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        prefetch_factor=2
    )

    start_time = time.time()
    count = 0
    for i, batch in enumerate(dataloader):
        count += 16
        if i >= 10: break # Process ~160 items
    duration = time.time() - start_time
    print(f"⏱️  160 items (DataLoader): {duration:.4f}s")
    print(f"⚡ Speed: {count/duration:.2f} items/s")

    # Benchmark Model Forward Pass
    print("\n🚀 Benchmarking MMRecModel Forward Pass...")
    from mm_rec.model import MMRecModel
    
    # Tiny config matching stage1_gpu.json
    model = MMRecModel(
        vocab_size=1000, # Dummy
        model_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=256,
        use_hem=True # Default as used in training
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️  Device: {device}")
    model.to(device)
    model.eval()
    
    model.eval()
    
    # Optional: torch.compile
    if hasattr(torch, "compile"):
        print("🚀 Compiling model with torch.compile...")
        try:
             # Using max-autotune for highest speed if stable
             model = torch.compile(model, mode="reduce-overhead")
             print("✅ Model compiled.")
        except Exception as e:
             print(f"⚠️ torch.compile failed: {e}")
             
    # Create dummy batch
    batch_size = 16
    seq_len = 256
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Warmup
    print("🔥 Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    # Benchmark
    print("⏱️  Running 10 forward passes...")
    start_time = time.time()
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
                
    duration = time.time() - start_time
    avg_fs = 10 / duration
    print(f"⏱️  10 passes: {duration:.4f}s")
    print(f"⚡ Speed: {avg_fs:.2f} it/s")

if __name__ == "__main__":
    benchmark()
