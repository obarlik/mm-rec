#!/usr/bin/env python3
"""
OpenAI-Compatible Training Script
Trains MM-Rec model with OpenAI chat format
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

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
    print(f"📝 Creating sample chat data ({num_samples} samples)...")
    
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
        # Add variation
        if i > 0:
            sample["messages"][-1]["content"] += f" (Example {i+1})"
        all_samples.append(sample)
    
    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Created {num_samples} samples: {output_path}")
    return output_path


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
    parser = argparse.ArgumentParser(description='OpenAI-compatible training for MM-Rec')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=100256, help='Vocabulary size (GPT-4: 100256)')
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    # Training config
    parser.add_argument('--data_file', type=str, default='./data/chat_data.jsonl')
    parser.add_argument('--create_sample_data', action='store_true', help='Create sample chat data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of sample conversations')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length (CPU-friendly)')
    parser.add_argument('--max_steps', type=int, default=100, help='Max training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_openai')
    parser.add_argument('--checkpoint_interval', type=int, default=25)
    
    # Tokenizer config
    parser.add_argument('--model_name', type=str, default='gpt-4', help='OpenAI model name')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Create sample data if requested
    if args.create_sample_data or not Path(args.data_file).exists():
        print("\n📝 Creating sample chat data...")
        create_sample_chat_data(args.data_file, args.num_samples)
    
    # Load data
    print(f"\n📦 Loading chat data from {args.data_file}...")
    conversations = load_chat_data(args.data_file)
    print(f"✅ Loaded {len(conversations)} conversations")
    
    if len(conversations) == 0:
        print("❌ No conversations found! Use --create_sample_data to create sample data.")
        return 1
    
    # Initialize tokenizer
    print(f"\n🔤 Initializing tokenizer ({args.model_name})...")
    try:
        tokenizer = get_tokenizer(model_name=args.model_name, vocab_size=args.vocab_size)
        print(f"✅ Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Install tiktoken: pip install tiktoken")
        return 1
    
    # Initialize model
    print(f"\n🤖 Initializing MM-Rec 100M model...")
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    
    print(f"✅ Model initialized ({model.get_num_params():,} parameters)")
    
    # Initialize trainer
    print(f"\n🎓 Initializing SFT Trainer...")
    config = SFTConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        only_predict_assistant=True
    )
    trainer = SFTTrainer(model, tokenizer, config)
    print("✅ Trainer initialized")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    # Training loop
    print(f"\n🚀 Starting training ({args.max_steps} steps)...")
    print("="*80)
    
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(range(args.max_steps), desc="Training")
    
    for step in pbar:
        # Sample random conversation
        conv_idx = step % len(conversations)
        messages_data = conversations[conv_idx]
        
        # Convert to ChatMessage objects
        messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages_data
        ]
        
        # Training step
        try:
            metrics = trainer.train_step(messages, optimizer, device)
            scheduler.step()
            
            total_loss += metrics['loss']
            avg_loss = total_loss / (step + 1)
            
            pbar.set_postfix({
                'loss': f'{metrics["loss"]:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'ppl': f'{metrics.get("perplexity", 0):.2f}',
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
                
                print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        
        except Exception as e:
            print(f"\n⚠️ Error at step {step}: {e}")
            continue
    
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {total_loss / args.max_steps:.4f}")
    print(f"   Checkpoints: {args.checkpoint_dir}")
    
    # Test inference
    print(f"\n🧪 Testing Chat Completion API...")
    api = ChatCompletionAPI(model, tokenizer)
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
    
    try:
        response = api.create(
            messages=test_messages,
            max_tokens=50,
            temperature=0.7,
            device=device
        )
        print(f"✅ Inference test successful!")
        print(f"   Response: {response['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"⚠️ Inference test failed: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

