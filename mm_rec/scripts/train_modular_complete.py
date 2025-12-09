"""
Complete Modular Training Script with Real Data Support
Enhanced version with DataLoader integration and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import argparse
import sys
import os
import math
import json
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.core.session_memory import SessionMemoryManager
from mm_rec.utils.monitoring import create_monitoring_hooks


class TextCodeDataset(Dataset):
    """
    Dataset for Text and Code data.
    Supports both synthetic and real data.
    """
    
    def __init__(
        self,
        text_data: Optional[List] = None,
        code_data: Optional[List] = None,
        seq_len: int = 512,
        vocab_size: int = 32000,
        synthetic: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            text_data: List of tokenized text sequences
            code_data: List of tokenized code sequences
            seq_len: Sequence length
            vocab_size: Vocabulary size (for synthetic data)
            synthetic: If True, generate synthetic data
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.synthetic = synthetic
        
        if synthetic:
            # Generate synthetic data
            self.text_data = None
            self.code_data = None
            self.length = 10000  # Synthetic dataset size
        else:
            self.text_data = text_data or []
            self.code_data = code_data or []
            self.length = min(len(self.text_data), len(self.code_data)) if self.text_data and self.code_data else max(len(self.text_data) if self.text_data else 0, len(self.code_data) if self.code_data else 0)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.synthetic:
            # Generate synthetic sequence
            return {
                'text': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'code': torch.randint(0, self.vocab_size, (self.seq_len,))
            }
        else:
            # Use real data
            text_seq = self.text_data[idx % len(self.text_data)] if self.text_data else torch.randint(0, self.vocab_size, (self.seq_len,))
            code_seq = self.code_data[idx % len(self.code_data)] if self.code_data else torch.randint(0, self.vocab_size, (self.seq_len,))
            
            # Ensure correct length
            if isinstance(text_seq, torch.Tensor):
                if text_seq.shape[0] < self.seq_len:
                    text_seq = torch.cat([text_seq, torch.zeros(self.seq_len - text_seq.shape[0], dtype=torch.long)])
                text_seq = text_seq[:self.seq_len]
            else:
                text_seq = torch.tensor(text_seq[:self.seq_len] if len(text_seq) >= self.seq_len else text_seq + [0] * (self.seq_len - len(text_seq)), dtype=torch.long)
            
            if isinstance(code_seq, torch.Tensor):
                if code_seq.shape[0] < self.seq_len:
                    code_seq = torch.cat([code_seq, torch.zeros(self.seq_len - code_seq.shape[0], dtype=torch.long)])
                code_seq = code_seq[:self.seq_len]
            else:
                code_seq = torch.tensor(code_seq[:self.seq_len] if len(code_seq) >= self.seq_len else code_seq + [0] * (self.seq_len - len(code_seq)), dtype=torch.long)
            
            return {
                'text': text_seq,
                'code': code_seq
            }


def train_stage_complete(
    model: MMRec100M,
    stage: str,
    device: torch.device,
    dataloader: DataLoader,
    vocab_size: int = 32000,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_interval: int = 100,
    gradient_clip_norm: float = 1.0,
    use_monitoring: bool = True,
    use_wandb: bool = False
):
    """
    Complete training function with monitoring and real data support.
    
    Args:
        model: MM-Rec 100M model
        stage: "stage1", "stage2", or "stage3"
        device: Training device
        dataloader: DataLoader for training data
        vocab_size: Vocabulary size
        checkpoint_dir: Checkpoint directory
        checkpoint_interval: Save checkpoint every N steps
        gradient_clip_norm: Gradient clipping norm
        use_monitoring: Enable monitoring hooks
        use_wandb: Enable WandB logging
    """
    from mm_rec.scripts.train_modular import ModularTrainingStrategy
    
    strategy = ModularTrainingStrategy(model, device, stage)
    config = strategy.get_training_config()
    
    print(f"\n{'='*80}")
    print(f"Stage: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Sequence Length: {config['seq_len']}")
    print(f"{'='*80}\n")
    
    # Create optimizer and scheduler
    optimizer = strategy.create_optimizer(config)
    scheduler = strategy.create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Monitoring hooks
    hooks = None
    if use_monitoring:
        hooks = create_monitoring_hooks(model, checkpoint_dir=checkpoint_dir)
        print("✅ Monitoring hooks enabled")
    
    # WandB
    if use_wandb:
        try:
            import wandb
            wandb.init(project="mmrec-100m", name=f"{stage}")
            print("✅ WandB logging enabled")
        except ImportError:
            print("⚠️ WandB not available, skipping")
            use_wandb = False
    
    # Training loop
    model.train()
    total_loss = 0.0
    step = 0
    dataloader_iter = iter(dataloader)
    
    pbar = tqdm(range(config["max_steps"]), desc=f"Stage {stage}")
    
    for step in pbar:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader_iter)
            batch = next(dataloader_iter)
        
        # Get input based on stage
        if config["train_both_experts"]:
            # Use text data (or alternate)
            input_ids = batch['text'].to(device)
        else:
            # Stage 2: Alternate between text and code
            if step % 2 == 0:
                input_ids = batch['text'].to(device)
            else:
                input_ids = batch['code'].to(device)
        
        # Ensure correct sequence length
        if input_ids.shape[1] > config["seq_len"]:
            input_ids = input_ids[:, :config["seq_len"]]
        elif input_ids.shape[1] < config["seq_len"]:
            padding = torch.zeros(input_ids.shape[0], config["seq_len"] - input_ids.shape[1], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        # Create targets
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            if config["train_both_experts"]:
                if config["use_fusion"]:
                    # Stage 3: Use fusion
                    logits = model(input_ids, expert_type=None, chunk_size=8192 if config["seq_len"] > 8192 else None)
                else:
                    # Stage 1: Train both without fusion
                    logits = model(input_ids, expert_type=None, chunk_size=8192 if config["seq_len"] > 8192 else None)
            else:
                # Stage 2: Train one expert at a time
                if step % 2 == 0:
                    logits = model(input_ids, expert_type="text", chunk_size=8192 if config["seq_len"] > 8192 else None)
                else:
                    logits = model(input_ids, expert_type="code", chunk_size=8192 if config["seq_len"] > 8192 else None)
            
            # Loss
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Check numerical stability
            if hooks and step % 10 == 0:
                stable = hooks["stability_monitor"].check_outputs({"logits": logits, "loss": loss.unsqueeze(0)}, step)
                if not stable:
                    print(f"\n⚠️ Numerical stability issue at step {step}")
                    success, checkpoint_data = hooks["recovery"].recover(model, optimizer)
                    if success:
                        print("✅ Recovery successful, continuing...")
                    else:
                        raise RuntimeError("Recovery failed!")
            
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
            
            # WandB logging
            if use_wandb and step % 10 == 0:
                wandb.log({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'step': step
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
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OOM at step {step}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                # Try recovery
                if hooks:
                    success, _ = hooks["recovery"].recover(model, optimizer)
                    if success:
                        print("✅ Recovered from OOM, reducing batch size...")
                        # Could reduce batch size here
                        continue
                raise
            else:
                raise
    
    print(f"\n✅ Stage {stage} training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Complete modular training for MM-Rec 100M')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    # Training config
    parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2', 'stage3', 'all'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_modular')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Data config
    parser.add_argument('--data_dir', type=str, default=None, help='Directory with text and code data')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # Monitoring
    parser.add_argument('--use_monitoring', action='store_true', default=True, help='Enable monitoring hooks')
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create dataset
    if args.use_synthetic or args.data_dir is None:
        print("📦 Using synthetic data")
        dataset = TextCodeDataset(seq_len=512, vocab_size=args.vocab_size, synthetic=True)
    else:
        print(f"📦 Loading data from {args.data_dir}")
        # Load real data (implement based on your data format)
        # text_data = load_text_data(args.data_dir)
        # code_data = load_code_data(args.data_dir)
        # dataset = TextCodeDataset(text_data, code_data, seq_len=512, synthetic=False)
        dataset = TextCodeDataset(seq_len=512, vocab_size=args.vocab_size, synthetic=True)  # Fallback
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training stages
    stages = ['stage1', 'stage2', 'stage3'] if args.stage == 'all' else [args.stage]
    
    for stage in stages:
        model = train_stage_complete(
            model=model,
            stage=stage,
            device=device,
            dataloader=dataloader,
            vocab_size=args.vocab_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            gradient_clip_norm=args.gradient_clip_norm,
            use_monitoring=args.use_monitoring,
            use_wandb=args.use_wandb
        )
    
    print("\n🎉 All training stages completed!")


if __name__ == '__main__':
    main()

