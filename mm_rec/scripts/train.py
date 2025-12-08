"""
MM-Rec Training Script - Production-Ready
Comprehensive training infrastructure with checkpointing, metrics, and scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Tuple
import argparse
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import math

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel


class TokenizerSimulator:
    """
    Simulates a real tokenizer for data loading.
    Mimics the structure of real tokenizers (e.g., HuggingFace tokenizers).
    """
    
    def __init__(self, vocab_size: int):
        """
        Initialize tokenizer simulator.
        
        Args:
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
        # Special tokens (similar to real tokenizers)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text: str, max_length: Optional[int] = None) -> list:
        """
        Simulate encoding text to token IDs.
        In real implementation, this would use actual tokenizer.
        
        Args:
            text: Input text (not used in simulation)
            max_length: Maximum sequence length
        
        Returns:
            List of token IDs
        """
        # For simulation, generate random token IDs
        # In real implementation, this would tokenize actual text
        seq_len = max_length if max_length else 128
        return torch.randint(
            self.bos_token_id + 1,  # Skip special tokens
            self.vocab_size,
            size=(seq_len,)
        ).tolist()
    
    def decode(self, token_ids: list) -> str:
        """
        Simulate decoding token IDs to text.
        In real implementation, this would use actual tokenizer.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text (simulated)
        """
        # For simulation, return placeholder
        return f"[Decoded: {len(token_ids)} tokens]"


class DataLoaderSimulator:
    """
    Simulates a real data loader for training.
    Mimics the structure of PyTorch DataLoader with tokenizer integration.
    """
    
    def __init__(
        self,
        tokenizer: TokenizerSimulator,
        batch_size: int,
        seq_len: int,
        num_batches: int,
        device: torch.device
    ):
        """
        Initialize data loader simulator.
        
        Args:
            tokenizer: TokenizerSimulator instance
            batch_size: Batch size
            seq_len: Sequence length
            num_batches: Number of batches to generate
            device: Device to create tensors on
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.device = device
    
    def __iter__(self):
        """Iterator for data loader."""
        for _ in range(self.num_batches):
            # Simulate tokenizer.encode() calls
            batch_input_ids = []
            batch_target_ids = []
            
            for _ in range(self.batch_size):
                # Simulate encoding text (in real case, would use actual text)
                input_ids = self.tokenizer.encode("", max_length=self.seq_len)
                
                # For language modeling, targets are input shifted by 1
                # In real case, this would be handled by tokenizer/collate function
                target_ids = input_ids[1:] + [self.tokenizer.eos_token_id]
                
                batch_input_ids.append(input_ids)
                batch_target_ids.append(target_ids)
            
            # Convert to tensors (similar to DataLoader collate function)
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
            target_ids = torch.tensor(batch_target_ids, dtype=torch.long, device=self.device)
            
            yield input_ids, target_ids
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches


class TrainingMetrics:
    """Track and compute training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.losses = []
        self.perplexities = []
        self.step = 0
    
    def update(self, loss: float):
        """
        Update metrics with new loss value.
        
        Args:
            loss: Loss value
        """
        self.losses.append(loss)
        # Perplexity = exp(loss)
        perplexity = math.exp(loss) if loss < 10 else float('inf')  # Prevent overflow
        self.perplexities.append(perplexity)
        self.step += 1
    
    def get_avg_loss(self, window: int = 100) -> float:
        """
        Get average loss over recent window.
        
        Args:
            window: Number of recent steps to average
        
        Returns:
            Average loss
        """
        if not self.losses:
            return 0.0
        recent_losses = self.losses[-window:]
        return sum(recent_losses) / len(recent_losses)
    
    def get_avg_perplexity(self, window: int = 100) -> float:
        """
        Get average perplexity over recent window.
        
        Args:
            window: Number of recent steps to average
        
        Returns:
            Average perplexity
        """
        if not self.perplexities:
            return float('inf')
        recent_ppls = [p for p in self.perplexities[-window:] if p != float('inf')]
        if not recent_ppls:
            return float('inf')
        return sum(recent_ppls) / len(recent_ppls)
    
    def get_current_loss(self) -> float:
        """Get most recent loss."""
        return self.losses[-1] if self.losses else 0.0
    
    def get_current_perplexity(self) -> float:
        """Get most recent perplexity."""
        return self.perplexities[-1] if self.perplexities else float('inf')


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    step: int,
    metrics: TrainingMetrics,
    checkpoint_dir: Path,
    is_best: bool = False
) -> Path:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        step: Current training step
        metrics: Training metrics
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint filename
    if is_best:
        checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    # Prepare checkpoint data
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {
            'losses': metrics.losses,
            'perplexities': metrics.perplexities,
            'step': metrics.step
        }
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata
    metadata = {
        'step': step,
        'checkpoint_path': str(checkpoint_path),
        'avg_loss': metrics.get_avg_loss(),
        'avg_perplexity': metrics.get_avg_perplexity(),
        'is_best': is_best
    }
    
    metadata_path = checkpoint_dir / f"checkpoint_step_{step}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, TrainingMetrics]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Tuple of (step, metrics)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore metrics
    metrics = TrainingMetrics()
    if 'metrics' in checkpoint:
        metrics.losses = checkpoint['metrics'].get('losses', [])
        metrics.perplexities = checkpoint['metrics'].get('perplexities', [])
        metrics.step = checkpoint['metrics'].get('step', 0)
    
    step = checkpoint.get('step', 0)
    
    print(f"✅ Checkpoint loaded: step={step}, avg_loss={metrics.get_avg_loss():.4f}")
    return step, metrics


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory to search
    
    Returns:
        Path to latest checkpoint or None
    """
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Sort by step number (extract from filename)
    def get_step(path: Path) -> int:
        try:
            # Extract step from "checkpoint_step_{step}.pt"
            return int(path.stem.split('_')[-1])
        except:
            return 0
    
    checkpoint_files.sort(key=get_step, reverse=True)
    return checkpoint_files[0]


def create_scheduler(
    optimizer: optim.Optimizer,
    num_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1
) -> optim.lr_scheduler._LRScheduler:
    """
    Create cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        num_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR
    
    Returns:
        Learning rate scheduler
    """
    # Warmup: linear increase from 0 to initial_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of initial LR
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing: decrease from initial_lr to min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_steps - warmup_steps,
        eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
    )
    
    # Sequential: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


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
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
    targets_flat = target_ids.view(-1)  # [batch*seq_len]
    
    # Compute loss
    loss = criterion(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping (optional, helps with stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def train_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    data_loader: DataLoaderSimulator,
    num_steps: int,
    device: torch.device,
    checkpoint_dir: Path,
    checkpoint_interval: int = 100,
    resume_from: Optional[Path] = None
):
    """
    Main training loop with checkpointing and metrics.
    
    Args:
        model: MM-Rec model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        data_loader: Data loader
        num_steps: Number of training steps
        device: Device
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Save checkpoint every N steps
        resume_from: Optional checkpoint path to resume from
    """
    model.train()
    
    # Initialize metrics
    metrics = TrainingMetrics()
    start_step = 0
    
    # Resume from checkpoint if provided
    if resume_from is not None and resume_from.exists():
        start_step, metrics = load_checkpoint(resume_from, model, optimizer, scheduler)
        print(f"🔄 Resuming training from step {start_step}")
    else:
        # Try to find latest checkpoint
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            print(f"📂 Found latest checkpoint: {latest_checkpoint}")
            response = input("Resume from latest checkpoint? (y/n): ").strip().lower()
            if response == 'y':
                start_step, metrics = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
                print(f"🔄 Resuming training from step {start_step}")
    
    print(f"\n{'='*80}")
    print(f"Starting training loop...")
    print(f"  Steps: {start_step} to {num_steps} (total: {num_steps - start_step})")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Create progress bar
    pbar = tqdm(
        range(start_step, num_steps),
        desc="Training",
        unit="step",
        initial=start_step,
        total=num_steps
    )
    
    # Data loader iterator
    data_iter = iter(data_loader)
    
    best_loss = float('inf')
    
    for step in pbar:
        try:
            # Get next batch
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            # Restart data loader if exhausted
            data_iter = iter(data_loader)
            input_ids, target_ids = next(data_iter)
        
        # Training step
        loss = train_step(
            model=model,
            input_ids=input_ids,
            target_ids=target_ids,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Update metrics
        metrics.update(loss)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = metrics.get_avg_loss()
        avg_ppl = metrics.get_avg_perplexity()
        
        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'avg_loss': f"{avg_loss:.4f}",
            'ppl': f"{metrics.get_current_perplexity():.2f}",
            'avg_ppl': f"{avg_ppl:.2f}",
            'lr': f"{current_lr:.2e}"
        })
        
        # Save checkpoint
        if (step + 1) % checkpoint_interval == 0:
            is_best = loss < best_loss
            if is_best:
                best_loss = loss
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step + 1,
                metrics=metrics,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
    
    # Save final checkpoint
    print(f"\n💾 Saving final checkpoint...")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=num_steps,
        metrics=metrics,
        checkpoint_dir=checkpoint_dir,
        is_best=False
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {metrics.get_current_loss():.4f}")
    print(f"   Final perplexity: {metrics.get_current_perplexity():.2f}")
    print(f"   Average loss: {metrics.get_avg_loss():.4f}")
    print(f"   Average perplexity: {metrics.get_avg_perplexity():.2f}")


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train MM-Rec model (Production-Ready)')
    
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
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of training steps (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps for learning rate scheduler (default: 100)')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                       help='Save checkpoint every N steps (default: 100)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from (default: None)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if device.type == 'cuda':
        print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU device (training will be slower)")
    
    print("\n" + "="*80)
    print("MM-Rec Training Script - Production-Ready")
    print("="*80)
    print()
    
    # Initialize tokenizer simulator
    print("📝 Initializing tokenizer simulator...")
    tokenizer = TokenizerSimulator(vocab_size=args.vocab_size)
    print(f"   Vocabulary size: {args.vocab_size}")
    print()
    
    # Initialize data loader simulator
    print("📦 Initializing data loader simulator...")
    data_loader = DataLoaderSimulator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=args.num_steps * 2,  # More batches than steps (for iteration)
        device=device
    )
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Number of batches: {len(data_loader)}")
    print()
    
    # Initialize model
    print("🤖 Initializing model...")
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
    print(f"✅ Model initialized:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Model size: {num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()
    
    # Initialize optimizer
    print("⚙️  Initializing optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"   Optimizer: AdamW")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Weight decay: {args.weight_decay}")
    print()
    
    # Initialize scheduler
    print("📈 Initializing learning rate scheduler...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=0.1
    )
    print(f"   Scheduler: Cosine Annealing with Warmup")
    print(f"   Warmup steps: {args.warmup_steps}")
    print(f"   Total steps: {args.num_steps}")
    print()
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    print(f"📊 Loss function: CrossEntropyLoss")
    print()
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    print()
    
    # Resume from checkpoint if specified
    resume_from = None
    if args.resume_from:
        resume_from = Path(args.resume_from)
    
    # Training loop
    print("="*80)
    train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        data_loader=data_loader,
        num_steps=args.num_steps,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=resume_from
    )
    
    print("\n" + "="*80)
    print("✅ Training script completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
