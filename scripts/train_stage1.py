#!/usr/bin/env python3
"""
Stage 1: Tiny Model Training (6M parameters, ~1 hour)
Foundation training with basic patterns
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

print("="*80)
print("STAGE 1: Tiny Model Training")
print("="*80)
print(f"Start time: {datetime.now()}")

# Configuration
STAGE1_CONFIG = {
    'vocab_size': 100256,
    'model_dim': 64,
    'num_layers': 2,
    'num_heads': 2,
    'ffn_dim': 128,
    'use_sparse': False,  # Keep simple for Stage 1
    'use_dpg': False,
    'use_hem': False
}

NUM_EXAMPLES = 5000  # Subset for fast training
NUM_EPOCHS = 10
BATCH_SIZE = 1  # Process one at a time
LEARNING_RATE = 1e-3

# Setup
device = torch.device('cpu')
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
print("\n2. Creating tiny model...")
model = MMRecModel(**STAGE1_CONFIG).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")
print(f"   Model dim: {STAGE1_CONFIG['model_dim']}")
print(f"   Layers: {STAGE1_CONFIG['num_layers']}")

# Training setup
print("\n3. Setting up training...")
config = SFTConfig(max_length=256, label_smoothing=0.1)
trainer = SFTTrainer(model, tokenizer, config)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n4. Training...")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Examples per epoch: {len(conversations)}")
print(f"   Optimizer: Adam (lr={LEARNING_RATE})")
print(f"   Estimated time: ~1 hour")
print()

import time
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 80)
    
    epoch_losses = []
    epoch_start = time.time()
    last_update = time.time()
    
    for i, conv in enumerate(conversations):
        result = trainer.train_step(conv, optimizer, device, verbose=False)
        epoch_losses.append(result['loss'])
        
        # Progress every 10 seconds
        current_time = time.time()
        if current_time - last_update >= 10.0:
            recent_loss = sum(epoch_losses[-100:]) / min(100, len(epoch_losses[-100:]))
            pct = (i + 1) / len(conversations) * 100
            elapsed = current_time - epoch_start
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Estimate completion
            steps_per_sec = (i + 1) / elapsed
            remaining_steps = len(conversations) - (i + 1)
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_min = eta_seconds / 60
            
            print(f"  [{i+1}/{len(conversations)} {pct:.1f}%] loss={recent_loss:.4f} | "
                  f"Adam(lr={current_lr:.0e}) | {elapsed:.0f}s | ETA:{eta_min:.0f}m", flush=True)
            last_update = current_time
        
        # Also report every 500 steps
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

total_time = time.time() - start_time

# Save model
print("\n5. Saving model...")
os.makedirs("checkpoints/progressive", exist_ok=True)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': STAGE1_CONFIG,
    'final_loss': avg_loss,
    'final_ppl': ppl,
    'training_time': total_time
}

torch.save(checkpoint, "checkpoints/progressive/stage1_tiny.pt")
print(f"   Saved to: checkpoints/progressive/stage1_tiny.pt")

# Final summary
print("\n" + "="*80)
print("✅ STAGE 1 COMPLETE!")
print("="*80)
print(f"Final loss: {avg_loss:.4f}")
print(f"Final PPL: {ppl:.2f}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Avg time/epoch: {total_time/NUM_EPOCHS/60:.1f} minutes")
print()
print("Next: Run Stage 2 to expand model to 13M parameters")
print("="*80)
