#!/usr/bin/env python3
"""
MM-Rec Pre-training Script

Pre-trains the MM-Rec model on raw text data (not chat format).
This is the first stage where the model learns English language.

Usage:
    python mm_rec/scripts/pretrain.py \
        --data_dir ./data/pretrain \
        --model_name mmrec_100m \
        --max_steps 100000 \
        --batch_size 4 \
        --seq_len 2048
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import math
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer


class PreTrainingDataLoader:
    """Load raw text data for pre-training."""
    
    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        seq_len: int = 2048,
        batch_size: int = 4
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.texts = []
        self._load_data()
    
    def _load_data(self):
        """Load all text files."""
        print(f"📦 Loading data from {len(self.data_files)} files...")
        for file_path in self.data_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        # JSONL format: each line is a JSON object
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    # Extract text from various formats
                                    if isinstance(data, dict):
                                        text = data.get('text', '') or data.get('content', '')
                                    else:
                                        text = str(data)
                                    if text:
                                        self.texts.append(text)
                                except:
                                    pass
                    else:
                        # Plain text file
                        text = f.read()
                        # Split by paragraphs or sentences
                        paragraphs = text.split('\n\n')
                        self.texts.extend([p.strip() for p in paragraphs if p.strip()])
        
        print(f"✅ Loaded {len(self.texts)} text chunks")
    
    def get_batch(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get a batch of tokenized sequences."""
        if not self.texts:
            return None
        
        batch_inputs = []
        
        for _ in range(self.batch_size):
            # Sample random text
            text = self.texts[torch.randint(0, len(self.texts), (1,)).item()]
            
            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                max_length=self.seq_len + 1,  # +1 for target
                truncation=True,
                padding=False
            )
            
            if len(tokens) < 2:
                # Skip if too short, try another text
                continue
            
            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long)
            batch_inputs.append(input_ids)
        
        if not batch_inputs:
            return None
        
        # Pad to same length (use seq_len + 1 as max)
        max_len = min(max(len(ids) for ids in batch_inputs), self.seq_len + 1)
        padded_batch = []
        for ids in batch_inputs:
            if len(ids) > max_len:
                ids = ids[:max_len]  # Truncate if too long
            padding = torch.zeros(max_len - len(ids), dtype=torch.long)
            padded = torch.cat([ids, padding])
            padded_batch.append(padded)
        
        # Ensure we have exactly batch_size items
        if len(padded_batch) < self.batch_size:
            # If not enough items, pad with last item
            while len(padded_batch) < self.batch_size:
                if padded_batch:
                    padded_batch.append(padded_batch[-1].clone())
                else:
                    # If empty, return None
                    return None
        
        # Stack and ensure correct shape
        batch_tensor = torch.stack(padded_batch[:self.batch_size]).to(device)
        
        # Ensure all sequences have same length
        if batch_tensor.size(1) != self.seq_len + 1:
            # Truncate or pad to exact length
            current_len = batch_tensor.size(1)
            if current_len > self.seq_len + 1:
                batch_tensor = batch_tensor[:, :self.seq_len + 1]
            elif current_len < self.seq_len + 1:
                padding = torch.zeros(
                    self.batch_size, 
                    self.seq_len + 1 - current_len, 
                    dtype=batch_tensor.dtype, 
                    device=device
                )
                batch_tensor = torch.cat([batch_tensor, padding], dim=1)
        
        return batch_tensor


def compute_pretrain_loss(
    model: MMRec100M,
    input_ids: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute pre-training loss (next token prediction).
    
    Args:
        model: MM-Rec model
        input_ids: [batch, seq_len] input token IDs
        ignore_index: Index to ignore in loss (for padding)
    
    Returns:
        Loss tensor
    """
    batch_size, seq_len = input_ids.shape
    
    # Shift input for next token prediction
    if seq_len < 2:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    
    input_ids_input = input_ids[:, :-1]  # [batch, seq_len-1]
    input_ids_target = input_ids[:, 1:]  # [batch, seq_len-1]
    
    # Forward pass
    # MMRec100M returns only logits by default (return_memory=False)
    logits = model(input_ids_input)  # [batch, seq_len-1, vocab_size]
    
    # Verify shapes match
    if logits.shape[0] != input_ids_target.shape[0] or logits.shape[1] != input_ids_target.shape[1]:
        # Shape mismatch - return zero loss
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = input_ids_target.reshape(-1)
    
    # Create mask for padding (0 is typically padding token)
    mask = (targets_flat != ignore_index) & (targets_flat != 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    
    # Compute loss
    loss = nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


def main():
    parser = argparse.ArgumentParser(description="MM-Rec Pre-training Script")
    parser.add_argument("--data_dir", type=str, default="./data/pretrain",
                        help="Directory containing pre-training data")
    parser.add_argument("--model_name", type=str, default="mmrec_100m",
                        help="Model name")
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Maximum training steps (50K for real pre-training)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (4 for CPU, 8-16 for GPU)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length (2048 for real pre-training)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use Automatic Mixed Precision (AMP) for training")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_pretrain",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="Checkpoint interval (steps, 5000 for real pre-training)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint path")
    parser.add_argument("--expert_dim", type=int, default=256,
                        help="Expert dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ffn_dim", type=int, default=1024,
                        help="FFN dimension")
    parser.add_argument("--vocab_size", type=int, default=100277,
                        help="Vocabulary size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("MM-Rec Pre-training")
    print("="*80)
    print(f"🖥️  Device: {device}")
    print(f"📦 Data directory: {args.data_dir}")
    print(f"🤖 Model: {args.model_name}")
    print(f"📊 Steps: {args.max_steps}")
    print(f"📏 Sequence length: {args.seq_len}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Find data files
    data_dir = Path(args.data_dir)
    data_files = []
    if data_dir.exists():
        data_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.jsonl"))
    
    if not data_files:
        print(f"❌ No data files found in {args.data_dir}")
        print(f"💡 Please download pre-training data first:")
        print(f"   - WikiText-103")
        print(f"   - OpenWebText")
        print(f"   - C4 dataset")
        return 1
    
    print(f"✅ Found {len(data_files)} data files")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=args.vocab_size)
    print(f"✅ Tokenizer ready (vocab={tokenizer.vocab_size})")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    
    # Check C++ optimizations
    # On CPU, C++ extension is REQUIRED (not optional)
    cpp_available = False
    cpp_error = None
    
    try:
        import mm_rec_cpp_cpu
        cpp_available = True
        print(f"✅ C++ optimizations: AVAILABLE")
    except ImportError as e:
        # Try with explicit path and library preloading
        try:
            import sys
            import os
            import ctypes
            
            # Preload PyTorch libraries using ctypes (REQUIRED for libc10.so)
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib):
                # Preload libc10.so with RTLD_GLOBAL
                libc10_path = os.path.join(torch_lib, 'libc10.so')
                if os.path.exists(libc10_path):
                    try:
                        ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
                
                # Set LD_LIBRARY_PATH
                os.environ['LD_LIBRARY_PATH'] = torch_lib
            
            # Try loading from build path (script is in mm_rec/scripts/, so go up to mm_rec/cpp/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # From mm_rec/scripts/ to mm_rec/cpp/build/...
            cpp_build_path = os.path.join(script_dir, '../cpp/build/lib.linux-x86_64-cpython-312')
            cpp_build_path = os.path.abspath(cpp_build_path)
            
            if os.path.exists(cpp_build_path):
                sys.path.insert(0, cpp_build_path)
                import mm_rec_cpp_cpu
                cpp_available = True
                print(f"✅ C++ optimizations: LOADED from build path")
            else:
                cpp_error = f"Build path not found: {cpp_build_path}"
        except Exception as e:
            cpp_error = f"{type(e).__name__}: {str(e)}"
    
    # On CPU, C++ extension is REQUIRED
    if device.type == 'cpu' and not cpp_available:
        print(f"\n❌ CRITICAL ERROR: C++ extension is REQUIRED for CPU mode!")
        print(f"   Error: {cpp_error or 'Import failed'}")
        print(f"\n💡 Solutions:")
        print(f"   1. Rebuild C++ extension:")
        print(f"      cd mm_rec/cpp && python3 setup.py build_ext --inplace")
        print(f"   2. Check PyTorch installation")
        print(f"   3. Check library paths")
        print(f"\n❌ Pre-training cannot start without C++ extension on CPU!")
        return 1
    
    print(f"✅ Model initialized ({model.get_num_params():,} params)")
    
    # Initialize data loader
    print("\n📦 Initializing data loader...")
    data_loader = PreTrainingDataLoader(
        data_files=[str(f) for f in data_files],
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps - args.warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 Starting Pre-training")
    print("="*80)
    # cpp_available is defined above (line 299-305)
    # On CPU, C++ should be active (it's built for CPU)
    if device.type == 'cpu' and not cpp_available:
        # Try to load C++ extension with explicit path
        try:
            import sys
            import os
            cpp_build_path = os.path.join(os.path.dirname(__file__), '../../cpp/build/lib.linux-x86_64-cpython-312')
            if os.path.exists(cpp_build_path):
                sys.path.insert(0, cpp_build_path)
                import mm_rec_cpp_cpu
                cpp_available = True
                print(f"✅ C++ optimizations: LOADED (CPU mode)")
        except Exception as e:
            print(f"⚠️  C++ optimizations: Could not load ({e})")
    
    cpp_status_msg = "✅ ACTIVE" if cpp_available else "❌ INACTIVE (Python fallback)"
    print(f"⚙️  C++ Optimizations: {cpp_status_msg}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Steps: {args.max_steps}")
    print("="*80)
    
    # Mixed precision training (AMP) - must be defined before training loop
    scaler = None
    autocast_context = None
    use_cpu_amp = False
    
    if args.use_amp:
        if device.type == 'cuda':
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_context = autocast
            print(f"✅ Mixed Precision (AMP): ENABLED (GPU)")
        elif device.type == 'cpu':
            # CPU-specific mixed precision
            from ..core.cpu_amp import CPUScaler, CPUAutocast, convert_model_to_mixed_precision
            scaler = CPUScaler()
            autocast_context = CPUAutocast(dtype=torch.bfloat16)
            use_cpu_amp = True
            # Convert model to mixed precision (FP16/BF16 storage)
            model = convert_model_to_mixed_precision(model, dtype=torch.bfloat16)
            print(f"✅ Mixed Precision (AMP): ENABLED (CPU - BF16 storage, FP32 computation)")
            print(f"   Memory savings: ~50% (model weights in BF16)")
    
    model.train()
    
    # Gradient checkpointing
    if args.use_gradient_checkpointing:
        # Enable gradient checkpointing in model if supported
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        print(f"✅ Gradient Checkpointing: ENABLED")
    
    total_loss = 0.0
    start_time = time.time()
    start_step = 0
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if provided
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            print(f"\n📂 Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint.get('step', 0) + 1
            total_loss = checkpoint.get('avg_loss', 0.0) * start_step
            print(f"✅ Resumed from step {start_step}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    pbar = tqdm(range(start_step, args.max_steps), desc="Pre-training", ncols=100, initial=start_step)
    
    # Gradient accumulation
    accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = args.batch_size * accumulation_steps
    
    for step in pbar:
        # Get batch
        batch = data_loader.get_batch(device)
        if batch is None:
            continue
        
        # Forward pass (with AMP if enabled)
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                loss = compute_pretrain_loss(model, batch)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        else:
            loss = compute_pretrain_loss(model, batch)
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation: only update every N steps
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            if args.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics (unscale for display)
        loss_display = loss.item() * accumulation_steps  # Unscale for display
        total_loss += loss_display
        avg_loss = total_loss / (step - start_step + 1)
        
        # Update progress bar with detailed optimization status
        lr = scheduler.get_last_lr()[0]
        cpp_status = "✅C++" if cpp_available else "⚠️Py"
        amp_status = "AMP" if scaler is not None else ""
        acc_status = f"acc{accumulation_steps}" if accumulation_steps > 1 else ""
        ckpt_status = "CKPT" if args.use_gradient_checkpointing else ""
        
        # Build optimization status string
        opt_parts = [cpp_status]
        if amp_status:
            opt_parts.append(amp_status)
        if acc_status:
            opt_parts.append(acc_status)
        if ckpt_status:
            opt_parts.append(ckpt_status)
        opt_status = " ".join(opt_parts)
        
        pbar.set_postfix({
            'loss': f'{loss_display:.4f}',
            'avg': f'{avg_loss:.4f}',
            'lr': f'{lr:.2e}',
            'opt': opt_status,
            'eff_batch': effective_batch_size if accumulation_steps > 1 else ''
        })
        
        # Checkpoint
        if (step + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step,
                'loss': loss.item(),
                'avg_loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            pbar.write(f"💾 Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    
    # Final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': args.max_steps,
        'loss': loss.item(),
        'avg_loss': avg_loss,
        'args': vars(args)
    }, final_checkpoint)
    
    print("\n" + "="*80)
    print("✅ Pre-training completed!")
    print(f"📊 Total time: {total_time/3600:.2f} hours")
    print(f"📈 Final loss: {loss.item():.4f}")
    print(f"📈 Average loss: {avg_loss:.4f}")
    print(f"💾 Final checkpoint: {final_checkpoint}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

