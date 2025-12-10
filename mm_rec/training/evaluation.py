"""
Evaluation Metrics for MM-Rec Training
Kaliteli eğitim için evaluation metrikleri
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity (exp(loss))
    """
    if loss > 10.0:  # Prevent overflow
        return float('inf')
    return math.exp(loss)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] (with -100 for ignored tokens)
    
    Returns:
        Accuracy (0-1)
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Mask ignored tokens
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    use_uboo: bool = False,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device
        use_uboo: Whether to use UBÖO auxiliary loss
        max_batches: Maximum number of batches to evaluate (None = all)
    
    Returns:
        Dictionary with metrics: loss, perplexity, accuracy
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if use_uboo:
                logits, L_Aux = model(input_ids, return_auxiliary_loss=True)
            else:
                logits = model(input_ids, return_auxiliary_loss=False)
            
            # Loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Accuracy
            batch_accuracy = compute_accuracy(shift_logits, shift_labels)
            mask = (shift_labels != -100)
            batch_tokens = mask.sum().item()
            total_tokens += batch_tokens
            total_correct += batch_accuracy * batch_tokens
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'num_batches': num_batches
    }


def print_evaluation_metrics(metrics: Dict[str, float], prefix: str = "Validation"):
    """Print evaluation metrics in a formatted way"""
    print(f"\n{prefix} Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}" if metrics['perplexity'] != float('inf') else "  Perplexity: inf")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Batches: {metrics['num_batches']}")
    print()
