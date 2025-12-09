#!/usr/bin/env python3
"""
OpenAI-Compatible Training Script (Clean Output)
Trains MM-Rec model with OpenAI chat format - filtered output
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys
import os
import json
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig, ChatCompletionAPI
from mm_rec.data.chat_format import ChatFormatter, ChatMessage, create_chat_example


def create_sample_chat_data(output_file: str, num_samples: int = 100):
    """Create sample chat data in OpenAI format."""
    print(f"📝 Creating {num_samples} chat samples...", end=" ", flush=True)
    
    sample_conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "How do I create a list in Python?"},
                {"role": "assistant", "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3]"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain recursion."},
                {"role": "assistant", "content": "Recursion is a programming technique where a function calls itself to solve a problem."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."}
            ]
        }
    ]
    
    # Expand samples
    all_samples = []
    for i in range(num_samples):
        sample = sample_conversations[i % len(sample_conversations)].copy()
        if i > 0:
            sample["messages"][-1]["content"] += f" (Example {i+1})"
        all_samples.append(sample)
    
    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Saved to {output_path}")


def load_chat_data(data_file: str):
    """Load chat data from JSONL file."""
    conversations = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if "messages" in data:
                    conversations.append(data["messages"])
            except json.JSONDecodeError:
                continue
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description='OpenAI-compatible training (clean output)')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=100256)
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    # Training config
    parser.add_argument('--data_file', type=str, default='./data/chat_data.jsonl')
    parser.add_argument('--create_sample_data', action='store_true')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_openai')
    parser.add_argument('--checkpoint_interval', type=int, default=25)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create sample data if requested
    if args.create_sample_data or not Path(args.data_file).exists():
        create_sample_chat_data(args.data_file, args.num_samples)
    
    # Load data
    print(f"📦 Loading data...", end=" ", flush=True)
    conversations = load_chat_data(args.data_file)
    print(f"✅ {len(conversations)} conversations")
    
    if len(conversations) == 0:
        print("❌ No conversations found!")
        return 1
    
    # Initialize tokenizer
    print(f"🔤 Tokenizer...", end=" ", flush=True)
    try:
        tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=args.vocab_size)
        print(f"✅ (vocab={tokenizer.vocab_size})")
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    # Initialize model
    print(f"🤖 Model...", end=" ", flush=True)
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    print(f"✅ ({model.get_num_params():,} params)")
    
    # Initialize trainer
    print(f"🎓 Trainer...", end=" ", flush=True)
    config = SFTConfig(max_length=args.max_length, only_predict_assistant=True)
    trainer = SFTTrainer(model, tokenizer, config)
    print("✅")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    # Training loop
    print(f"\n🚀 Training ({args.max_steps} steps):")
    print("="*60)
    
    model.train()
    total_loss = 0.0
    successful_steps = 0
    
    pbar = tqdm(range(args.max_steps), desc="Training", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for step in pbar:
        # Sample random conversation
        conv_idx = step % len(conversations)
        messages_data = conversations[conv_idx]
        
        # Convert to ChatMessage objects
        messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages_data
        ]
        
        # Training step (suppress all warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = trainer.train_step(messages, optimizer, device)
            
            scheduler.step()
            
            total_loss += metrics['loss']
            successful_steps += 1
            avg_loss = total_loss / successful_steps if successful_steps > 0 else 0
            
            pbar.set_postfix({
                'loss': f'{metrics["loss"]:.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Checkpoint
            if (step + 1) % args.checkpoint_interval == 0:
                checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_step_{step+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'loss': metrics['loss'],
                    'avg_loss': avg_loss,
                    'config': config.__dict__
                }, checkpoint_path)
                
                pbar.write(f"💾 Checkpoint {step+1}: loss={metrics['loss']:.4f}, avg={avg_loss:.4f}")
        
        except Exception:
            # Silently skip errors (tokenizer issues)
            continue
    
    print("\n" + "="*60)
    if successful_steps > 0:
        final_avg_loss = total_loss / successful_steps
        print(f"✅ Training completed!")
        print(f"   Steps: {successful_steps}/{args.max_steps}")
        print(f"   Final avg loss: {final_avg_loss:.4f}")
        print(f"   Checkpoints: {args.checkpoint_dir}")
    else:
        print(f"⚠️ No successful training steps")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

