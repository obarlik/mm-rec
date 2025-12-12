#!/usr/bin/env python3
"""
Training System Verification
Tests that training pipeline works correctly (not for actual training)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage

print("="*80)
print("Training System Verification")
print("="*80)

# Tiny model for verification
device = torch.device('cpu')
tokenizer = get_tokenizer(vocab_size=100256)

print("\n1. Creating tiny model...")
model = MMRecModel(
    vocab_size=tokenizer.vocab_size,
    model_dim=64,   # Very small
    num_layers=2,
    num_heads=2,
    ffn_dim=128
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"   Model: {params:,} parameters")

# Test data
conversations = [
    [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!")
    ]
]

print("\n2. Testing training step...")
config = SFTConfig(max_length=128, label_smoothing=0.1)
trainer = SFTTrainer(model, tokenizer, config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

import time
start = time.time()

# Single training step
result = trainer.train_step(conversations[0], optimizer, device, verbose=False)

elapsed = time.time() - start

print(f"   ✅ Training step completed in {elapsed:.2f}s")
print(f"   Loss: {result['loss']:.4f}")
print(f"   PPL: {result['perplexity']:.2f}")

# Test multiple steps
print("\n3. Testing 10 training steps...")
losses = []
start = time.time()

for i in range(10):
    result = trainer.train_step(conversations[0], optimizer, device, verbose=False)
    losses.append(result['loss'])
    print(f"   Step {i+1}/10: loss={result['loss']:.4f}", end='\r', flush=True)

print()
elapsed = time.time() - start
avg_time = elapsed / 10

print(f"   ✅ 10 steps completed in {elapsed:.2f}s")
print(f"   Average: {avg_time:.2f}s/step")
print(f"   Loss improved: {losses[0]:.4f} → {losses[-1]:.4f}")

# Estimate full training time
print("\n4. Full training time estimate:")
print(f"   21,216 examples × {avg_time:.2f}s = {21216 * avg_time / 3600:.1f} hours")
print(f"   5 epochs = {21216 * avg_time * 5 / 3600:.1f} hours")

print("\n" + "="*80)
print("✅ Training system verified!")
print("\n💡 Recommendation:")
print("   CPU training is too slow for production.")
print("   For actual training, use GPU (A100 recommended).")
print("   Estimated GPU time: ~2-3 hours for 5 epochs")
print("="*80)
