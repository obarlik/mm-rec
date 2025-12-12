#!/usr/bin/env python3
"""
Quick training test with 100 examples to verify everything works
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage
import json

print("="*80)
print("Quick Training Test (100 examples)")
print("="*80)

# Load first 100 examples
with open('data/phase1/train.json', 'r') as f:
    data = json.load(f)[:100]

conversations = []
for item in data:
    messages = [ChatMessage(role=msg['role'], content=msg['content']) 
                for msg in item['conversations']]
    conversations.append(messages)

print(f"Loaded {len(conversations)} conversations")

# Small model
device = torch.device('cpu')
tokenizer = get_tokenizer(vocab_size=100256)

model = MMRecModel(
    vocab_size=tokenizer.vocab_size,
    model_dim=128,
    num_layers=2,
    num_heads=4,
    ffn_dim=512
).to(device)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Training
config = SFTConfig(max_length=256, label_smoothing=0.1)
trainer = SFTTrainer(model, tokenizer, config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining...")
for i, conv in enumerate(conversations):
    result = trainer.train_step(conv, optimizer, device, verbose=False)
    
    if (i + 1) % 10 == 0:
        print(f"  Step {i+1}/100: loss={result['loss']:.4f}")

print("\n✅ Test complete!")
