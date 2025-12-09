#!/usr/bin/env python3
"""
Test Training Script - Shows every step with detailed output
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import warnings
from pathlib import Path
import time

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage

def main():
    print("="*80)
    print("🧪 TEST TRAINING - Detailed Step-by-Step Output")
    print("="*80)
    
    device = torch.device('cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n📦 Loading data...", flush=True)
    chat_file = Path("./data/chat_data_real.jsonl")
    if not chat_file.exists():
        print(f"❌ Data file not found: {chat_file}")
        return 1
    
    conversations = []
    with open(chat_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "messages" in data:
                        conversations.append(data["messages"])
                except:
                    continue
    
    print(f"✅ Loaded {len(conversations)} conversations", flush=True)
    
    if len(conversations) == 0:
        print("❌ No conversations!")
        return 1
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...", flush=True)
    tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=100256)
    print(f"✅ Tokenizer ready (vocab={tokenizer.vocab_size})", flush=True)
    
    # Initialize model (smaller for test)
    print("\n🤖 Initializing model...", flush=True)
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,  # Smaller for test
        num_layers=4,    # Fewer layers
        num_heads=4,
        ffn_dim=512
    ).to(device)
    print(f"✅ Model ready ({model.get_num_params():,} params)", flush=True)
    
    # Initialize trainer
    print("\n🎓 Initializing trainer...", flush=True)
    config = SFTConfig(max_length=128, only_predict_assistant=True)
    trainer = SFTTrainer(model, tokenizer, config)
    print("✅ Trainer ready", flush=True)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training loop - VERY DETAILED
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING - Every step will be shown")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    model.train()
    max_steps = 10  # Just 10 steps for test
    
    for step in range(max_steps):
        print(f"\n{'='*80}")
        print(f"📌 STEP {step+1}/{max_steps}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Get conversation
        conv_idx = step % len(conversations)
        messages_data = conversations[conv_idx]
        
        print(f"📝 Using conversation {conv_idx+1}/{len(conversations)}", flush=True)
        
        # Convert to ChatMessage
        messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages_data
        ]
        
        print(f"💬 Messages: {len(messages)}", flush=True)
        for i, msg in enumerate(messages):
            print(f"   {i+1}. {msg.role}: {msg.content[:50]}...", flush=True)
        
        # Training step
        print(f"🔄 Running forward pass...", flush=True)
        step_start = time.time()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = trainer.train_step(messages, optimizer, device)
            
            step_time = time.time() - step_start
            
            loss = metrics['loss']
            perplexity = metrics.get('perplexity', 0)
            
            print(f"✅ Step {step+1} COMPLETED!", flush=True)
            print(f"   Loss: {loss:.4f}", flush=True)
            print(f"   Perplexity: {perplexity:.2f}", flush=True)
            print(f"   Time: {step_time:.2f}s", flush=True)
            print(f"   Speed: {1/step_time:.2f} steps/s", flush=True)
            
        except Exception as e:
            print(f"❌ ERROR at step {step+1}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        
        sys.stdout.flush()
    
    print("\n" + "="*80)
    print("✅ TEST TRAINING COMPLETED!")
    print("="*80)
    sys.stdout.flush()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

