"""
CPU-Optimized Training Script for MM-Rec 100M
Optimized for CPU training without GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.data.load_data import create_dataloader
from mm_rec.scripts.train_modular import ModularTrainingStrategy


def train_cpu(
    model: MMRec100M,
    stage: str,
    dataloader,
    vocab_size: int = 32000,
    checkpoint_dir: str = "./checkpoints_cpu",
    checkpoint_interval: int = 50,
    gradient_clip_norm: float = 1.0,
    use_cpu_optimizations: bool = True
):
    """
    CPU-optimized training function.
    
    Args:
        model: MM-Rec 100M model
        stage: Training stage
        dataloader: DataLoader instance
        vocab_size: Vocabulary size
        checkpoint_dir: Checkpoint directory
        checkpoint_interval: Save checkpoint every N steps
        gradient_clip_norm: Gradient clipping norm
        use_cpu_optimizations: Enable CPU-specific optimizations
    """
    device = torch.device('cpu')
    model = model.to(device)
    
    strategy = ModularTrainingStrategy(model, device, stage)
    config = strategy.get_training_config()
    
    print(f"\n{'='*80}")
    print(f"CPU Training - Stage: {config['name']}")
    print(f"Device: CPU")
    print(f"Sequence Length: {config['seq_len']}")
    print(f"{'='*80}\n")
    
    # CPU Optimizations
    if use_cpu_optimizations:
        # Use MKL for better CPU performance
        torch.set_num_threads(os.cpu_count())
        print(f"✅ Using {os.cpu_count()} CPU threads")
        
        # Disable CUDA operations
        torch.backends.cudnn.enabled = False
    
    # Create optimizer and scheduler
    optimizer = strategy.create_optimizer(config)
    scheduler = strategy.create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    total_loss = 0.0
    step = 0
    dataloader_iter = iter(dataloader)
    
    # Adjust max_steps for CPU (fewer steps, faster iteration)
    max_steps = min(config["max_steps"], 1000)  # Limit for CPU training
    print(f"📊 Training for {max_steps} steps (CPU-optimized)")
    
    pbar = tqdm(range(max_steps), desc=f"CPU Training - {stage}")
    
    for step in pbar:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Get input based on stage
        if config["train_both_experts"]:
            input_ids = batch['text'].to(device)
        else:
            if step % 2 == 0:
                input_ids = batch['text'].to(device)
            else:
                input_ids = batch['code'].to(device)
        
        # Ensure correct sequence length
        if input_ids.shape[1] > config["seq_len"]:
            input_ids = input_ids[:, :config["seq_len"]]
        elif input_ids.shape[1] < config["seq_len"]:
            padding = torch.zeros(input_ids.shape[0], config["seq_len"] - input_ids.shape[1], dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        # Create targets
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # CPU-friendly: Smaller sequences, no chunking overhead
            if config["train_both_experts"]:
                if config["use_fusion"]:
                    logits = model(input_ids, expert_type=None)
                else:
                    logits = model(input_ids, expert_type=None)
            else:
                if step % 2 == 0:
                    logits = model(input_ids, expert_type="text")
                else:
                    logits = model(input_ids, expert_type="code")
            
            # Loss
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ Numerical error at step {step}, skipping...")
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Checkpoint
            if (step + 1) % checkpoint_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{stage}_step_{step+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'stage': stage,
                    'config': config
                }, checkpoint_path)
                
                print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        
        except Exception as e:
            print(f"\n❌ Error at step {step}: {e}")
            continue
    
    print(f"\n✅ CPU training completed for stage {stage}!")
    return model


def main():
    parser = argparse.ArgumentParser(description='CPU-optimized training for MM-Rec 100M')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    # Training config
    parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2', 'stage3', 'all'])
    parser.add_argument('--batch_size', type=int, default=2, help='Smaller batch for CPU')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cpu')
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Data config
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--download_data', action='store_true', help='Download data automatically')
    parser.add_argument('--text_samples', type=int, default=500, help='Text samples to download')
    parser.add_argument('--code_samples', type=int, default=500, help='Code samples to download')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples per dataset')
    
    args = parser.parse_args()
    
    # Download data if requested
    if args.download_data:
        print("📥 Downloading data...")
        from mm_rec.data.download_data import DataDownloader
        downloader = DataDownloader(output_dir=args.data_dir)
        downloader.download_all(
            text_samples=args.text_samples,
            code_samples=args.code_samples
        )
        print("✅ Data download completed!")
    
    # Device
    device = torch.device('cpu')
    print(f"\n🖥️  Using device: {device}")
    print(f"   CPU threads: {os.cpu_count()}")
    
    # Create model
    model = MMRec100M(
        vocab_size=args.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    
    print(f"\n📊 Model Parameters: {model.get_num_params():,} ({model.get_num_params()/1e6:.2f}M)")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Checkpoint loaded")
    
    # Create DataLoader
    print("\n📦 Creating DataLoader...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        seq_len=512,  # CPU için küçük sekanslar
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        num_workers=0,  # CPU için 0
        synthetic=args.use_synthetic,
        max_samples=args.max_samples
    )
    print(f"✅ DataLoader created: {len(dataloader)} batches")
    
    # Training stages
    stages = ['stage1', 'stage2', 'stage3'] if args.stage == 'all' else [args.stage]
    
    for stage in stages:
        model = train_cpu(
            model=model,
            stage=stage,
            dataloader=dataloader,
            vocab_size=args.vocab_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            gradient_clip_norm=args.gradient_clip_norm
        )
    
    print("\n🎉 All CPU training stages completed!")
    print("\n💡 Note: CPU training is slower than GPU. For production training, use GPU.")


if __name__ == '__main__':
    main()

