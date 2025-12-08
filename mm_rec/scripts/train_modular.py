"""
Modular Training Script for MM-Rec 100M Model
Staged training: Local consistency → Global specialization → Fusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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


class ModularTrainingStrategy:
    """
    Modular training strategy with staged approach:
    1. Stage 1 (Local): MDI gates learn local consistency with short sequences
    2. Stage 2 (Global & Separate): Experts specialize on long sequences
    3. Stage 3 (Fusion): Fusion layer learns to combine experts
    """
    
    def __init__(
        self,
        model: MMRec100M,
        device: torch.device,
        stage: str = "stage1"
    ):
        self.model = model
        self.device = device
        self.stage = stage
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for current stage."""
        configs = {
            "stage1": {
                "name": "Local Consistency",
                "description": "MDI gates learn local consistency",
                "seq_len": 512,  # Short sequences
                "train_both_experts": True,
                "use_fusion": False,
                "learning_rate": 3e-4,
                "warmup_steps": 500,
                "max_steps": 5000
            },
            "stage2": {
                "name": "Global Specialization",
                "description": "Experts specialize on domain-specific long sequences",
                "seq_len": 8192,  # Long sequences
                "train_both_experts": False,  # Train separately
                "use_fusion": False,
                "learning_rate": 1e-4,
                "warmup_steps": 1000,
                "max_steps": 10000
            },
            "stage3": {
                "name": "Fusion Training",
                "description": "Fusion layer learns to combine experts",
                "seq_len": 4096,
                "train_both_experts": True,
                "use_fusion": True,
                "learning_rate": 5e-5,
                "warmup_steps": 500,
                "max_steps": 5000
            }
        }
        return configs.get(self.stage, configs["stage1"])
    
    def create_optimizer(self, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer for current stage."""
        # Stage 2: Train experts separately
        if self.stage == "stage2":
            # Freeze one expert, train the other
            for param in self.model.text_expert.parameters():
                param.requires_grad = True
            for param in self.model.code_expert.parameters():
                param.requires_grad = False  # Train text first, then code
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        return optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        config: Dict[str, Any]
    ) -> Any:
        """Create learning rate scheduler."""
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config["warmup_steps"]
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["max_steps"] - config["warmup_steps"],
            eta_min=config["learning_rate"] * 0.1
        )
        
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config["warmup_steps"]]
        )


def train_stage(
    model: MMRec100M,
    stage: str,
    device: torch.device,
    vocab_size: int = 32000,
    batch_size: int = 4,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_interval: int = 100,
    gradient_clip_norm: float = 1.0
):
    """
    Train model for a specific stage.
    
    Args:
        model: MM-Rec 100M model
        stage: "stage1", "stage2", or "stage3"
        device: Training device
        vocab_size: Vocabulary size
        batch_size: Batch size
        checkpoint_dir: Checkpoint directory
        checkpoint_interval: Save checkpoint every N steps
        gradient_clip_norm: Gradient clipping norm
    """
    strategy = ModularTrainingStrategy(model, device, stage)
    config = strategy.get_training_config()
    
    print(f"\n{'='*80}")
    print(f"Stage: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")
    
    # Create optimizer and scheduler
    optimizer = strategy.create_optimizer(config)
    scheduler = strategy.create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(range(config["max_steps"]), desc=f"Stage {stage}")
    
    for step in pbar:
        # Generate synthetic data (replace with real data loader)
        seq_len = config["seq_len"]
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        optimizer.zero_grad()
        
        if config["train_both_experts"]:
            if config["use_fusion"]:
                # Stage 3: Use fusion
                logits = model(input_ids, expert_type=None)
            else:
                # Stage 1: Train both without fusion
                logits = model(input_ids, expert_type=None)
        else:
            # Stage 2: Train one expert at a time
            if step % 2 == 0:
                # Train text expert
                logits = model(input_ids, expert_type="text")
            else:
                # Train code expert
                logits = model(input_ids, expert_type="code")
        
        # Loss
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        
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
                'stage': stage,
                'config': config
            }, checkpoint_path)
            
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
    
    print(f"\n✅ Stage {stage} training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Modular training for MM-Rec 100M')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--expert_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    
    # Training config
    parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2', 'stage3', 'all'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_modular')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--resume_from', type=str, default=None)
    
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
    
    # Training stages
    stages = ['stage1', 'stage2', 'stage3'] if args.stage == 'all' else [args.stage]
    
    for stage in stages:
        model = train_stage(
            model=model,
            stage=stage,
            device=device,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            gradient_clip_norm=args.gradient_clip_norm
        )
    
    print("\n🎉 All training stages completed!")


if __name__ == '__main__':
    main()

