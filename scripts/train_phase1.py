#!/usr/bin/env python3
"""
Phase 1 Training Script: Foundation Training
Simple Q&A and short conversations using curriculum learning
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from typing import List, Dict
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage


def load_conversations(filepath: str) -> List[List[ChatMessage]]:
    """Load conversations from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for item in data:
        messages = []
        for msg in item['conversations']:
            messages.append(ChatMessage(
                role=msg['role'],
                content=msg['content']
            ))
        conversations.append(messages)
    
    return conversations


def train_phase1(args):
    """Execute Phase 1 training."""
    
    print("="*80)
    print("MM-Rec Phase 1 Training: Foundation")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    print("\n📚 Loading datasets...")
    train_data = load_conversations(os.path.join(args.data_dir, 'train.json'))
    val_data = load_conversations(os.path.join(args.data_dir, 'val.json'))
    
    print(f"   Training: {len(train_data)} conversations")
    print(f"   Validation: {len(val_data)} conversations")
    
    # Tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = get_tokenizer(vocab_size=100256)
    
    # Model
    print("\n🧠 Initializing model...")
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        use_sparse=True,
        num_experts=16,
        use_dpg=True,
        use_hem=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training config
    print("\n⚙️  Training configuration...")
    config = SFTConfig(
        max_length=512,           # Short sequences for Phase 1
        label_smoothing=0.1,
        ignore_index=-100
    )
    
    # Training hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    
    print(f"   Max length: {config.max_length}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")
    
    # Trainer
    trainer = SFTTrainer(model, tokenizer, config)
    
    # Optimizer
    print("\n🎯 Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        # Training
        model.train()
        train_losses = []
        
        # Shared state for async progress monitoring
        progress_state = {
            'current_step': 0,
            'total_steps': len(train_data),
            'recent_losses': [],
            'running': True
        }
        
        # Async progress monitor thread
        def progress_monitor():
            """Background thread that prints progress every 5 seconds."""
            while progress_state['running']:
                time.sleep(5)
                if progress_state['current_step'] > 0 and progress_state['recent_losses']:
                    step = progress_state['current_step']
                    total = progress_state['total_steps']
                    recent_loss = sum(progress_state['recent_losses'][-10:]) / min(10, len(progress_state['recent_losses']))
                    ppl = torch.exp(torch.tensor(recent_loss)).item()
                    pct = (step / total) * 100
                    
                    # Compact one-line update
                    print(f"\r   [{step}/{total} {pct:.1f}%] loss={recent_loss:.4f} ppl={ppl:.1f}    ", 
                          end='', flush=True)
        
        # Start progress monitor
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        # Training loop (no blocking progress code)
        for i, conversation in enumerate(train_data):
            # Training step
            result = trainer.train_step(
                conversation,
                optimizer,
                device,
                verbose=False
            )
            
            # Update shared state (thread-safe for simple updates)
            train_losses.append(result['loss'])
            progress_state['current_step'] = i + 1
            progress_state['recent_losses'] = train_losses
            
            # Checkpoint every 100 steps
            if (i + 1) % 100 == 0:
                print()  # New line
                avg_loss_100 = sum(train_losses[-100:]) / 100
                print(f"   ✅ Step {i+1}: avg_loss={avg_loss_100:.4f}")
        
        # Stop progress monitor
        progress_state['running'] = False
        monitor_thread.join(timeout=1)
        print()  # Final newline
        
        # Epoch stats
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"\n   Training loss: {avg_train_loss:.4f}")
        
        # Validation
        print("\n   Validating...")
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for conversation in val_data[:min(100, len(val_data))]:  # Sample for speed
                # Prepare input
                input_ids, attention_mask, labels = trainer.prepare_chat_input(
                    conversation,
                    device
                )
                
                # Forward
                logits = model(input_ids)
                
                # Loss
                loss = trainer.compute_loss(logits, labels, attention_mask)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        
        print(f"   Validation loss: {avg_val_loss:.4f}")
        print(f"   Validation PPL: {val_ppl:.2f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(checkpoint_dir, 'model.pt'))
            
            print(f"   ✅ Saved checkpoint (best val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/3)")
        
        # Early stopping
        if patience_counter >= 3:
            print("\n⚠️  Early stopping triggered")
            break
    
    # Final save
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    final_dir = os.path.join(args.output_dir, "phase1_final")
    os.makedirs(final_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_val_loss
    }, os.path.join(final_dir, 'model.pt'))
    
    print(f"\n✅ Model saved to: {final_dir}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final validation PPL: {val_ppl:.2f}")
    print(f"\nEnd time: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training")
    
    # Data
    parser.add_argument("--data-dir", default="data/phase1", help="Data directory")
    parser.add_argument("--output-dir", default="checkpoints/phase1", help="Output directory")
    
    # Model
    parser.add_argument("--model-dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ffn-dim", type=int, default=2048, help="FFN dimension")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    train_phase1(args)


if __name__ == "__main__":
    main()
