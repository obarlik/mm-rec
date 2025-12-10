#!/usr/bin/env python3
"""
MM-Rec Expert Fine-tuning Script
Pretrained base model'i uzmanlık alanlarıyla fine-tune eder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel


def finetune_expert(
    checkpoint_path: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 4,
    seq_len: int = 512,
    learning_rate: float = 1e-5,  # Fine-tuning için daha düşük LR
    expert_name: str = "expert",
    device: Optional[torch.device] = None
):
    """
    Pretrained model'i uzmanlık alanıyla fine-tune et.
    
    Args:
        checkpoint_path: Pretrained model checkpoint path
        output_dir: Output directory
        num_epochs: Epoch sayısı
        batch_size: Batch size
        seq_len: Sequence length
        learning_rate: Learning rate (fine-tuning için düşük)
        expert_name: Expert model adı
        device: Device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 Device: {device}")
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    print()
    
    # Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Model oluştur
    model = MMRecModel(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📊 Model Parameters: {model.get_num_params():,}")
    print(f"📋 Expert: {expert_name}")
    print()
    
    # Fine-tuning için optimizer (sadece belirli layer'ları train et)
    # Örnek: Sadece son N layer'ı train et
    trainable_params = []
    for name, param in model.named_parameters():
        # Son 2 layer'ı train et, diğerlerini freeze et
        if 'blocks.22' in name or 'blocks.23' in name or 'lm_head' in name:
            trainable_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print(f"🔓 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print()
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1)
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Data loader (expert-specific data)
    # TODO: Gerçek expert data loader implementasyonu
    from mm_rec.scripts.train_base_model import SimpleDataLoader
    dataloader = SimpleDataLoader(
        vocab_size=model_config['vocab_size'],
        batch_size=batch_size,
        seq_len=min(seq_len, model_config.get('max_seq_len', 2048)),
        num_batches=50,  # Fine-tuning için daha az batch
        device=device
    )
    
    # Training loop
    print("🚀 Fine-tuning started...")
    print()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            logits = model(input_ids, return_auxiliary_loss=False)
            
            # Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
        print()
    
    # Save expert model
    expert_dir = Path(output_dir) / expert_name
    expert_dir.mkdir(parents=True, exist_ok=True)
    expert_path = expert_dir / "expert_checkpoint.pt"
    
    torch.save({
        'expert_name': expert_name,
        'base_checkpoint': checkpoint_path,
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'optimizer_state_dict': optimizer.state_dict(),
    }, expert_path)
    
    print(f"✅ Fine-tuning completed!")
    print(f"📂 Expert model saved: {expert_path}")
    
    return model, expert_path


def main():
    parser = argparse.ArgumentParser(description='MM-Rec Expert Fine-tuning')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Pretrained model checkpoint path')
    parser.add_argument('--output-dir', type=str, default='experts',
                       help='Output directory')
    parser.add_argument('--expert-name', type=str, required=True,
                       help='Expert model name (e.g., medical, code, math)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    finetune_expert(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        expert_name=args.expert_name
    )


if __name__ == "__main__":
    main()
