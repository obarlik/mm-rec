"""
MM-Rec Training Script
Basic training loop for MM-Rec model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_rec.model import MMRecModel


def create_synthetic_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to create tensors on
    
    Returns:
        Tuple of (input_ids, target_ids)
    """
    # Generate random input token IDs
    input_ids = torch.randint(
        0, vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long
    )
    
    # Target IDs are input IDs shifted by 1 (language modeling)
    # For simplicity, use same as input (in practice, shift by 1)
    target_ids = input_ids.clone()
    
    return input_ids, target_ids


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Perform a single training step.
    
    Args:
        model: MM-Rec model
        input_ids: Input token IDs [batch, seq_len]
        target_ids: Target token IDs [batch, seq_len]
        optimizer: Optimizer
        criterion: Loss function
        device: Device
    
    Returns:
        Loss value
    """
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(input_ids)  # [batch, seq_len, vocab_size]
    
    # Reshape logits and targets for loss computation
    # CrossEntropyLoss expects: [batch*seq_len, vocab_size] and [batch*seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
    targets_flat = target_ids.view(-1)  # [batch*seq_len]
    
    # Compute loss
    loss = criterion(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def train_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    print_interval: int = 10
):
    """
    Main training loop.
    
    Args:
        model: MM-Rec model
        optimizer: Optimizer
        criterion: Loss function
        num_steps: Number of training steps
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device
        print_interval: Print loss every N steps
    """
    model.train()
    
    print(f"Starting training loop...")
    print(f"  Steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Device: {device}")
    print()
    
    for step in range(num_steps):
        # Create synthetic data
        input_ids, target_ids = create_synthetic_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device
        )
        
        # Training step
        loss = train_step(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Print progress
        if (step + 1) % print_interval == 0:
            print(f"Step {step + 1}/{num_steps} | Loss: {loss:.4f}")
    
    print(f"\nTraining completed!")


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train MM-Rec model')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size (default: 1000)')
    parser.add_argument('--model_dim', type=int, default=512,
                       help='Model dimension (default: 512)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--mem_dim', type=int, default=256,
                       help='Memory dimension (default: 256)')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length (default: 128)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of training steps (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    
    # Other arguments
    parser.add_argument('--print_interval', type=int, default=10,
                       help='Print loss every N steps (default: 10)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    print("\n" + "="*60)
    print("MM-Rec Training Script")
    print("="*60)
    print()
    
    # Initialize model
    print("Initializing model...")
    model = MMRecModel(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_memories=1,
        mem_dim=args.mem_dim,
        max_seq_len=args.max_seq_len,
        dropout=0.1
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model size: {num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()
    
    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print()
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: CrossEntropyLoss")
    print()
    
    # Training loop
    print("="*60)
    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
        print_interval=args.print_interval
    )
    
    print("\n" + "="*60)
    print("Training script completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

