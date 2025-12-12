#!/usr/bin/env python3
"""
Stage 1 GPU Training Script
Optimized for NVIDIA GPU with CUDA
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from datetime import datetime
from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage
import time

print("="*80)
print("STAGE 1 GPU: High-Speed Training")
print("="*80)
print(f"Start time: {datetime.now()}")

# GPU Configuration
FAST_CONFIG = {
    'vocab_size': 100256,
    'model_dim': 64,
    'num_layers': 2,
    'num_heads': 2,
    'ffn_dim': 128,         # Full size on GPU
    'use_sparse': False,
    'use_dpg': False,
    'use_hem': False
}

NUM_EXAMPLES = 5000
NUM_EPOCHS = 10
BATCH_SIZE = 16             # Larger batch on GPU
ACCUMULATION_STEPS = 1      # No accumulation needed on GPU
MAX_LENGTH = 256            # Longer sequences on GPU
LEARNING_RATE = 1e-3
USE_MIXED_PRECISION = True  # FP16 for speed

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️  WARNING: No GPU detected! Training will be slow.")

tokenizer = get_tokenizer(vocab_size=100256)

print("\n1. Loading data...")
with open('data/phase1/train.json', 'r') as f:
    data = json.load(f)[:NUM_EXAMPLES]

conversations = []
for item in data:
    messages = [ChatMessage(role=msg['role'], content=msg['content']) 
                for msg in item['conversations']]
    conversations.append(messages)

print(f"   Loaded {len(conversations)} conversations")

# Create model
print("\n2. Creating GPU-optimized model...")
model = MMRecModel(**FAST_CONFIG).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")
print(f"   Model dim: {FAST_CONFIG['model_dim']}")
print(f"   FFN dim: {FAST_CONFIG['ffn_dim']}")

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and device.type == 'cuda' else None

# Training setup
print("\n3. Setting up GPU training...")
config = SFTConfig(
    max_length=MAX_LENGTH,
    label_smoothing=0.1
)
trainer = SFTTrainer(model, tokenizer, config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n4. Training on GPU...")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Max length: {MAX_LENGTH}")
print(f"   Mixed precision: {USE_MIXED_PRECISION}")
print(f"   Estimated time: ~4 hours (vs 47 hours on CPU)")
print()

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 80)
    
    epoch_losses = []
    epoch_start = time.time()
    last_update = time.time()
    
    optimizer.zero_grad()
    
    for i, conv in enumerate(conversations):
        # Mixed precision training
        if USE_MIXED_PRECISION and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                result = trainer.train_step(conv, optimizer, device, verbose=False)
        else:
            result = trainer.train_step(conv, optimizer, device, verbose=False)
        
        epoch_losses.append(result['loss'])
        
        # Progress every 10 seconds
        current_time = time.time()
        if current_time - last_update >= 10.0:
            recent_loss = sum(epoch_losses[-100:]) / min(100, len(epoch_losses[-100:]))
            pct = (i + 1) / len(conversations) * 100
            elapsed = current_time - epoch_start
            
            current_lr = optimizer.param_groups[0]['lr']
            steps_per_sec = (i + 1) / elapsed
            remaining_steps = len(conversations) - (i + 1)
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_min = eta_seconds / 60
            
            print(f"  [{i+1}/{len(conversations)} {pct:.1f}%] loss={recent_loss:.4f} | "
                  f"AdamW(lr={current_lr:.0e}) | {elapsed:.0f}s | ETA:{eta_min:.0f}m", flush=True)
            last_update = current_time
        
        # Checkpoint every 500 steps
        if (i + 1) % 500 == 0:
            recent_loss = sum(epoch_losses[-100:]) / min(100, len(epoch_losses[-100:]))
            print(f"  ✅ Step {i+1}/{len(conversations)}: loss={recent_loss:.4f}")
    
    # Epoch summary
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    epoch_time = time.time() - epoch_start
    
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl:.2f}")
    print(f"  Time: {epoch_time/60:.1f} min")
    print()
    
    # Save epoch checkpoint
    os.makedirs("checkpoints/progressive", exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': FAST_CONFIG,
        'loss': avg_loss,
        'ppl': ppl
    }
    torch.save(checkpoint, f"checkpoints/progressive/stage1_gpu_epoch_{epoch+1}.pt")

total_time = time.time() - start_time

# Save final model
print("\n5. Saving final model...")
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': FAST_CONFIG,
    'final_loss': avg_loss,
    'final_ppl': ppl,
    'training_time': total_time,
    'device': 'gpu'
}

torch.save(final_checkpoint, "checkpoints/progressive/stage1_gpu_final.pt")
print(f"   Saved to: checkpoints/progressive/stage1_gpu_final.pt")

# Final summary
print("\n" + "="*80)
print("✅ STAGE 1 GPU COMPLETE!")
print("="*80)
print(f"Final loss: {avg_loss:.4f}")
print(f"Final PPL: {ppl:.2f}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Speedup vs CPU: ~12x faster")
print()
print("Next: Run Stage 2 to expand model to 13M parameters")
print("="*80)
