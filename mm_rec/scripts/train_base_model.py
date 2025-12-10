#!/usr/bin/env python3
"""
MM-Rec Base Model Training Script
En küçük temel modeli eğitir, progressive upscaling için hazırlar
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
import math
from typing import Optional, Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.configs.base_model_configs import (
    TINY_BASE_CONFIG,
    MINI_BASE_CONFIG,
    SMALL_BASE_CONFIG,
    BASE_BASE_CONFIG,
    get_config_by_name,
    PROGRESSIVE_TRAINING_SEQUENCE
)
from mm_rec.utils.model_upscaling import upscale_model


class SimpleDataLoader:
    """Basit data loader (gerçek data yerine simüle edilmiş)"""
    
    def __init__(self, vocab_size: int, batch_size: int, seq_len: int, num_batches: int, device: torch.device):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.device = device
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        # Simüle edilmiş data
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        # Labels = input_ids shifted by 1
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[:, -1] = -100  # Ignore last token
        
        self.current_batch += 1
        return {'input_ids': input_ids, 'labels': labels}
    
    def __len__(self):
        return self.num_batches


def train_step(
    model: MMRecModel,
    batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_uboo: bool = False
) -> Dict[str, float]:
    """Tek bir training step"""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    if use_uboo:
        logits, L_Aux = model(input_ids, return_auxiliary_loss=True)
    else:
        logits = model(input_ids, return_auxiliary_loss=False)
        L_Aux = None
    
    # Loss calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Add auxiliary loss if UBÖO is enabled
    if L_Aux is not None:
        total_loss = loss + L_Aux
    else:
        total_loss = loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'total_loss': total_loss.item(),
        'auxiliary_loss': L_Aux.item() if L_Aux is not None else 0.0
    }


def train_base_model(
    config_name: str = "tiny",
    output_dir: str = "checkpoints",
    num_epochs: int = 10,
    batch_size: int = 4,
    seq_len: int = 512,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    resume_from: Optional[str] = None
):
    """
    Base model eğitimi.
    
    Args:
        config_name: Model konfigürasyonu adı (tiny, mini, small, base, medium, large, 7b)
        output_dir: Checkpoint kayıt dizini
        num_epochs: Epoch sayısı
        batch_size: Batch size
        seq_len: Sequence length
        learning_rate: Learning rate
        warmup_steps: Warmup step sayısı
        max_steps: Maximum step sayısı (None ise num_epochs kullanılır)
        device: Device (None ise otomatik)
        resume_from: Resume edilecek checkpoint path
    """
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 Device: {device}")
    print(f"📋 Config: {config_name}")
    print()
    
    # Config
    config = get_config_by_name(config_name)
    if config is None:
        raise ValueError(f"Unknown config: {config_name}")
    
    print(f"📐 Model Configuration:")
    print(f"   Vocab Size: {config.vocab_size:,}")
    print(f"   Model Dim: {config.model_dim}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Heads: {config.num_heads}")
    print(f"   HEM: {config.use_hem}")
    print(f"   DPG: {config.use_dpg}")
    print(f"   UBÖO: {config.use_uboo}")
    print()
    
    # Model
    if resume_from:
        print(f"📂 Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model = MMRecModel(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('step', 0)
    else:
        model = MMRecModel(
            vocab_size=config.vocab_size,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_memories=config.num_memories,
            mem_dim=config.mem_dim,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_hem=config.use_hem,
            use_dpg=config.use_dpg,
            use_uboo=config.use_uboo,
            dpg_rank=config.dpg_rank,
            lambda_P=config.lambda_P
        ).to(device)
        start_epoch = 0
        start_step = 0
    
    num_params = model.get_num_params()
    print(f"📊 Model Parameters: {num_params:,}")
    print(f"💾 Memory (FP16): {num_params * 2 / (1024**2):.2f} MB")
    print()
    
    # Data loader
    num_batches_per_epoch = 100  # Simüle edilmiş
    dataloader = SimpleDataLoader(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        seq_len=min(seq_len, config.max_seq_len),
        num_batches=num_batches_per_epoch,
        device=device
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Scheduler
    if max_steps is None:
        max_steps = num_epochs * len(dataloader)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=learning_rate * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    print("🚀 Training started...")
    print()
    
    global_step = start_step
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            metrics = train_step(model, batch, optimizer, criterion, device, use_uboo=config.use_uboo)
            scheduler.step()
            
            epoch_losses.append(metrics['loss'])
            global_step += 1
            
            # Progress bar update
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Checkpoint (her 100 step'te bir)
            if global_step % 100 == 0:
                checkpoint_dir = Path(output_dir) / config_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'model_config': {
                        'vocab_size': config.vocab_size,
                        'model_dim': config.model_dim,
                        'num_layers': config.num_layers,
                        'num_heads': config.num_heads,
                        'num_memories': config.num_memories,
                        'mem_dim': config.mem_dim,
                        'ffn_dim': config.ffn_dim,
                        'max_seq_len': config.max_seq_len,
                        'dropout': config.dropout,
                        'use_hem': config.use_hem,
                        'use_dpg': config.use_dpg,
                        'use_uboo': config.use_uboo,
                        'dpg_rank': config.dpg_rank,
                        'lambda_P': config.lambda_P
                    },
                    'loss': metrics['loss'],
                }, checkpoint_path)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
        print()
    
    # Final checkpoint
    final_checkpoint_path = Path(output_dir) / config_name / "final_checkpoint.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': num_epochs - 1,
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'vocab_size': config.vocab_size,
            'model_dim': config.model_dim,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'num_memories': config.num_memories,
            'mem_dim': config.mem_dim,
            'ffn_dim': config.ffn_dim,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout,
            'use_hem': config.use_hem,
            'use_dpg': config.use_dpg,
            'use_uboo': config.use_uboo,
            'dpg_rank': config.dpg_rank,
            'lambda_P': config.lambda_P
        },
    }, final_checkpoint_path)
    
    print(f"✅ Training completed!")
    print(f"📂 Final checkpoint saved: {final_checkpoint_path}")
    print()
    
    return model, final_checkpoint_path


def progressive_train(
    start_config_name: str = "tiny",
    end_config_name: str = "7b",
    output_dir: str = "checkpoints",
    epochs_per_stage: int = 5,
    **train_kwargs
):
    """
    Progressive training: Küçük modelden büyük modele adım adım eğitim.
    
    Args:
        start_config_name: Başlangıç konfigürasyonu
        end_config_name: Bitiş konfigürasyonu
        output_dir: Checkpoint dizini
        epochs_per_stage: Her stage için epoch sayısı
        **train_kwargs: train_base_model için ek parametreler
    """
    start_config = get_config_by_name(start_config_name)
    end_config = get_config_by_name(end_config_name)
    
    if start_config is None or end_config is None:
        raise ValueError(f"Invalid config names: {start_config_name}, {end_config_name}")
    
    # Progressive sequence bul
    start_idx = PROGRESSIVE_TRAINING_SEQUENCE.index(start_config)
    end_idx = PROGRESSIVE_TRAINING_SEQUENCE.index(end_config)
    
    if start_idx >= end_idx:
        raise ValueError(f"Start config must be smaller than end config")
    
    sequence = PROGRESSIVE_TRAINING_SEQUENCE[start_idx:end_idx+1]
    
    print("=" * 60)
    print("🚀 Progressive Training Başlatılıyor")
    print("=" * 60)
    print()
    print(f"📋 Training Sequence:")
    for i, config in enumerate(sequence):
        print(f"   {i+1}. {config.name} ({config.model_dim}D, {config.num_layers}L)")
    print()
    
    current_model = None
    current_checkpoint = None
    
    for stage_idx, config in enumerate(sequence):
        print("=" * 60)
        print(f"📦 Stage {stage_idx + 1}/{len(sequence)}: {config.name}")
        print("=" * 60)
        print()
        
        if current_model is None:
            # İlk stage: Sıfırdan eğit
            print("🆕 Starting from scratch...")
            current_model, current_checkpoint = train_base_model(
                config_name=config.name,
                output_dir=output_dir,
                num_epochs=epochs_per_stage,
                resume_from=None,
                **train_kwargs
            )
        else:
            # Sonraki stage: Upscale ve eğit
            print("🔄 Upscaling model...")
            current_model = upscale_model(
                current_model,
                config,
                device=current_model.parameters().__iter__().__next__().device,
                verbose=True
            )
            print()
            
            # Upscaled model'i eğit
            print("📚 Training upscaled model...")
            current_model, current_checkpoint = train_base_model(
                config_name=config.name,
                output_dir=output_dir,
                num_epochs=epochs_per_stage,
                resume_from=None,
                **train_kwargs
            )
            # Model'i checkpoint'ten yükle (train_base_model yeni model oluşturuyor)
            checkpoint = torch.load(current_checkpoint, map_location=current_model.parameters().__iter__().__next__().device)
            current_model.load_state_dict(checkpoint['model_state_dict'])
        
        print()
        print(f"✅ Stage {stage_idx + 1} completed!")
        print(f"📂 Checkpoint: {current_checkpoint}")
        print()
    
    print("=" * 60)
    print("🎉 Progressive Training Tamamlandı!")
    print("=" * 60)
    print()
    print(f"📂 Final Model: {current_checkpoint}")
    
    return current_model, current_checkpoint


def main():
    parser = argparse.ArgumentParser(description='MM-Rec Base Model Training')
    parser.add_argument('--config', type=str, default='tiny', 
                       choices=['tiny', 'mini', 'small', 'base', 'medium', 'large', '7b'],
                       help='Model configuration')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--progressive', action='store_true',
                       help='Progressive training (tiny -> 7b)')
    parser.add_argument('--start-config', type=str, default='tiny',
                       help='Start config for progressive training')
    parser.add_argument('--end-config', type=str, default='7b',
                       help='End config for progressive training')
    parser.add_argument('--epochs-per-stage', type=int, default=5,
                       help='Epochs per stage (for progressive training)')
    
    args = parser.parse_args()
    
    if args.progressive:
        progressive_train(
            start_config_name=args.start_config,
            end_config_name=args.end_config,
            output_dir=args.output_dir,
            epochs_per_stage=args.epochs_per_stage,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps
        )
    else:
        train_base_model(
            config_name=args.config,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            resume_from=args.resume_from
        )


if __name__ == "__main__":
    main()
