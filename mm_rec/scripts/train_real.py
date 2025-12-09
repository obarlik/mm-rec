#!/usr/bin/env python3
"""
Real Training Script for MM-Rec 100M
Full-scale training with real data, proper metrics, and production settings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import sys
import os
import json
import warnings
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatFormatter, ChatMessage
from mm_rec.data.download_data import DataDownloader


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.losses = []
        self.perplexities = []
        self.learning_rates = []
        self.timestamps = []
        self.step_times = []
    
    def update(self, loss, perplexity, lr, step_time):
        self.losses.append(loss)
        self.perplexities.append(perplexity)
        self.learning_rates.append(lr)
        self.timestamps.append(time.time())
        self.step_times.append(step_time)
    
    def get_summary(self):
        if not self.losses:
            return {}
        
        return {
            'total_steps': len(self.losses),
            'avg_loss': sum(self.losses) / len(self.losses),
            'min_loss': min(self.losses),
            'max_loss': max(self.losses),
            'final_loss': self.losses[-1],
            'avg_perplexity': sum(self.perplexities) / len(self.perplexities),
            'avg_step_time': sum(self.step_times) / len(self.step_times),
            'total_time': sum(self.step_times)
        }
    
    def save(self, filepath):
        """Save metrics to JSON."""
        data = {
            'losses': self.losses,
            'perplexities': self.perplexities,
            'learning_rates': self.learning_rates,
            'step_times': self.step_times,
            'summary': self.get_summary()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def download_real_data(output_dir: str, text_samples: int = 5000, code_samples: int = 5000):
    """Download real training data from internet."""
    print(f"\n📥 Downloading real training data...")
    print(f"   Text samples: {text_samples}")
    print(f"   Code samples: {code_samples}")
    
    downloader = DataDownloader(output_dir=output_dir)
    downloader.download_all(
        text_samples=text_samples,
        code_samples=code_samples,
        use_wikitext=True,
        use_code_datasets=True
    )
    
    print("✅ Data download completed!")


def load_chat_data(data_file: str):
    """Load chat data from JSONL file."""
    conversations = []
    
    print(f"📦 Loading data from {data_file}...", end=" ", flush=True)
    
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
    
    print(f"✅ {len(conversations)} conversations")
    return conversations


def create_chat_data_from_text(text_file: str, code_file: str, output_file: str, max_samples: int = 10000):
    """Convert text/code data to chat format using proper converter."""
    from mm_rec.data.convert_to_chat import convert_all_to_chat
    
    print(f"\n🔄 Converting to chat format...")
    
    data_dir = Path(text_file).parent.parent
    num_convs = convert_all_to_chat(
        data_dir=str(data_dir),
        output_file=output_file,
        max_samples=max_samples,
        text_ratio=0.6
    )
    
    # Load created conversations
    conversations = []
    if Path(output_file).exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        if "messages" in conv:
                            conversations.append(conv["messages"])
                    except:
                        continue
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description='Real training for MM-Rec 100M')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=100256)
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    # Training config
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--chat_data_file', type=str, default='./data/chat_data.jsonl')
    parser.add_argument('--download_data', action='store_true', help='Download real data from internet')
    parser.add_argument('--text_samples', type=int, default=5000, help='Text samples to download')
    parser.add_argument('--code_samples', type=int, default=5000, help='Code samples to download')
    parser.add_argument('--convert_to_chat', action='store_true', help='Convert text/code to chat format')
    parser.add_argument('--max_chat_samples', type=int, default=10000, help='Max chat conversations')
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=10000, help='Real training: 10K+ steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_real')
    parser.add_argument('--checkpoint_interval', type=int, default=500)
    parser.add_argument('--save_metrics', action='store_true', help='Save training metrics')
    parser.add_argument('--metrics_file', type=str, default='./training_metrics.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("MM-Rec 100M - Real Training")
    print("="*80)
    print(f"🖥️  Device: {device}")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Download real data if requested
    if args.download_data:
        download_real_data(args.data_dir, args.text_samples, args.code_samples)
    
    # Convert to chat format if requested
    if args.convert_to_chat:
        text_file = Path(args.data_dir) / "text" / "wikitext.jsonl"
        code_file = Path(args.data_dir) / "code" / "code.jsonl"
        create_chat_data_from_text(str(text_file), str(code_file), args.chat_data_file, args.max_chat_samples)
    
    # Load chat data
    if not Path(args.chat_data_file).exists():
        print(f"\n❌ Chat data file not found: {args.chat_data_file}")
        print("   Use --download_data and --convert_to_chat to create data")
        return 1
    
    conversations = load_chat_data(args.chat_data_file)
    
    if len(conversations) == 0:
        print("❌ No conversations found!")
        return 1
    
    # Initialize tokenizer
    print(f"\n🔤 Initializing tokenizer...", end=" ", flush=True)
    try:
        tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=args.vocab_size)
        print(f"✅ (vocab={tokenizer.vocab_size})")
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    # Initialize model
    print(f"🤖 Initializing model...", end=" ", flush=True)
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    print(f"✅ ({model.get_num_params():,} params)")
    
    # Initialize trainer
    print(f"🎓 Initializing trainer...", end=" ", flush=True)
    config = SFTConfig(max_length=args.max_length, only_predict_assistant=True)
    trainer = SFTTrainer(model, tokenizer, config)
    print("✅")
    
    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Scheduler with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])
    
    # Metrics
    metrics = TrainingMetrics()
    
    # Training loop
    print(f"\n🚀 Starting REAL training:")
    print(f"   Steps: {args.max_steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max length: {args.max_length}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Warmup steps: {args.warmup_steps}")
    print("="*80)
    
    model.train()
    start_time = time.time()
    last_print_time = time.time()
    print_interval = 5.0  # Print status every 5 seconds
    
    # Enhanced progress bar with more info
    pbar = tqdm(range(args.max_steps), desc="🚀 Training", ncols=120, 
                bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
                miniters=1, mininterval=1.0)
    
    print("\n" + "="*80)
    print("📊 Training Started - Real-time Updates")
    print("="*80 + "\n")
    
    # Force flush output
    sys.stdout.flush()
    
    step_count = 0
    for step in pbar:
        step_count += 1
        step_start = time.time()
        
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
            # Print step start (every step for visibility)
            if step_count <= 5 or step_count % 10 == 0:
                print(f"\n🔄 Step {step_count}/{args.max_steps} - Processing...", flush=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                step_metrics = trainer.train_step(messages, optimizer, device)
            
            scheduler.step()
            
            step_time = time.time() - step_start
            loss = step_metrics['loss']
            perplexity = step_metrics.get('perplexity', 0)
            lr = scheduler.get_last_lr()[0]
            
            metrics.update(loss, perplexity, lr, step_time)
            
            # ALWAYS print step completion
            print(f"✅ Step {step_count} completed: loss={loss:.4f}, time={step_time:.2f}s", flush=True)
            
            # Update progress bar with detailed info
            avg_loss = metrics.get_summary()['avg_loss']
            steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            elapsed_time = time.time() - start_time
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.1f}',
                'lr': f'{lr:.2e}',
                'spd': f'{steps_per_sec:.1f}/s'
            })
            
            # Print detailed status every N seconds
            current_time = time.time()
            if current_time - last_print_time >= print_interval or step == 0 or (step + 1) % 10 == 0:
                elapsed_str = f"{elapsed_time/60:.1f}m" if elapsed_time > 60 else f"{elapsed_time:.1f}s"
                eta_seconds = (args.max_steps - step - 1) * step_time
                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.1f}s"
                
                status_line = (
                    f"📈 Step {step+1}/{args.max_steps} | "
                    f"Loss: {loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"PPL: {perplexity:.1f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"Time: {elapsed_str} (ETA: {eta_str})"
                )
                pbar.write(status_line)
                last_print_time = current_time
            
            # Checkpoint with detailed info
            if (step + 1) % args.checkpoint_interval == 0:
                checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_step_{step+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'loss': loss,
                    'metrics': metrics.get_summary(),
                    'config': config.__dict__,
                    'args': vars(args)
                }, checkpoint_path)
                
                summary = metrics.get_summary()
                checkpoint_msg = (
                    f"\n💾 Checkpoint {step+1} saved!\n"
                    f"   Loss: {loss:.4f} | Avg Loss: {summary['avg_loss']:.4f} | "
                    f"Min Loss: {summary['min_loss']:.4f}\n"
                    f"   Avg PPL: {summary['avg_perplexity']:.2f} | "
                    f"Steps/sec: {1/summary['avg_step_time']:.2f}\n"
                    f"   File: {checkpoint_path}\n"
                )
                pbar.write(checkpoint_msg)
        
        except Exception as e:
            # Show error but continue
            error_msg = str(e)[:100]
            pbar.write(f"⚠️ Step {step+1} error: {error_msg}")
            continue
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    summary = metrics.get_summary()
    print("✅ Training completed!")
    print(f"\n📊 Summary:")
    print(f"   Total steps: {summary['total_steps']}")
    print(f"   Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"   Avg step time: {summary['avg_step_time']:.3f}s")
    print(f"   Steps/sec: {1/summary['avg_step_time']:.2f}")
    print(f"\n📈 Metrics:")
    print(f"   Final loss: {summary['final_loss']:.4f}")
    print(f"   Avg loss: {summary['avg_loss']:.4f}")
    print(f"   Min loss: {summary['min_loss']:.4f}")
    print(f"   Avg perplexity: {summary['avg_perplexity']:.2f}")
    print(f"\n💾 Checkpoints: {args.checkpoint_dir}")
    
    if args.save_metrics:
        metrics.save(args.metrics_file)
        print(f"📊 Metrics saved: {args.metrics_file}")
    
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

