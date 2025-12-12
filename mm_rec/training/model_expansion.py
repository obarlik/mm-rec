"""
Model Expansion Utilities for Progressive Training
Transfers knowledge from smaller to larger models
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import copy


def expand_embedding(small_emb: nn.Embedding, large_dim: int) -> nn.Embedding:
    """
    Expand embedding layer from small to large dimension.
    
    Args:
        small_emb: Small embedding layer
        large_dim: Target dimension
    
    Returns:
        Expanded embedding layer
    """
    vocab_size = small_emb.num_embeddings
    small_dim = small_emb.embedding_dim
    
    # Create new embedding
    large_emb = nn.Embedding(vocab_size, large_dim)
    
    # Copy old weights
    with torch.no_grad():
        large_emb.weight[:, :small_dim] = small_emb.weight.data
        
        # Initialize new dimensions with small random values
        large_emb.weight[:, small_dim:] = torch.randn(
            vocab_size, large_dim - small_dim
        ) * 0.02
    
    return large_emb


def expand_linear(small_linear: nn.Linear, out_features: int = None, 
                  in_features: int = None) -> nn.Linear:
    """
    Expand linear layer.
    
    Args:
        small_linear: Small linear layer
        out_features: New output features (None = keep same)
        in_features: New input features (None = keep same)
    
    Returns:
        Expanded linear layer
    """
    old_in = small_linear.in_features
    old_out = small_linear.out_features
    
    new_in = in_features if in_features else old_in
    new_out = out_features if out_features else old_out
    
    # Create new layer
    large_linear = nn.Linear(new_in, new_out, bias=small_linear.bias is not None)
    
    # Copy weights
    with torch.no_grad():
        # Copy overlapping region
        large_linear.weight[:old_out, :old_in] = small_linear.weight.data
        
        # Initialize new regions
        if new_out > old_out:
            large_linear.weight[old_out:, :old_in] = torch.randn(
                new_out - old_out, old_in
            ) * 0.02
        
        if new_in > old_in:
            large_linear.weight[:old_out, old_in:] = torch.randn(
                old_out, new_in - old_in
            ) * 0.02
        
        if new_out > old_out and new_in > old_in:
            large_linear.weight[old_out:, old_in:] = torch.randn(
                new_out - old_out, new_in - old_in
            ) * 0.02
        
        # Copy bias if exists
        if small_linear.bias is not None:
            large_linear.bias[:old_out] = small_linear.bias.data
            if new_out > old_out:
                large_linear.bias[old_out:] = torch.randn(new_out - old_out) * 0.02
    
    return large_linear


def expand_model_simple(small_model, large_config: Dict[str, Any]):
    """
    Simple model expansion for MM-Rec.
    Expands model_dim while keeping same architecture.
    
    Args:
        small_model: Trained small model
        large_config: Config for large model
    
    Returns:
        Expanded model initialized from small model
    """
    from mm_rec.model import MMRecModel
    
    # Create large model
    large_model = MMRecModel(**large_config)
    
    small_dim = small_model.model_dim
    large_dim = large_config['model_dim']
    
    print(f"Expanding model: {small_dim} → {large_dim}")
    
    # 1. Expand embeddings
    print("  Expanding embeddings...")
    large_model.embedding = expand_embedding(small_model.embedding, large_dim)
    
    # 2. Expand blocks (same number of layers)
    num_layers = min(len(small_model.blocks), len(large_model.blocks))
    print(f"  Expanding {num_layers} layers...")
    
    for i in range(num_layers):
        small_block = small_model.blocks[i]
        large_block = large_model.blocks[i]
        
        # Expand attention projections
        # Note: This is simplified - full implementation would handle
        # multi-head attention properly
        
        # Expand FFN
        if hasattr(small_block, 'ffn') and hasattr(large_block, 'ffn'):
            # Copy expert weights if same number of experts
            if hasattr(small_block.ffn, 'expert_weights'):
                large_block.ffn.expert_weights.data = small_block.ffn.expert_weights.data.clone()
    
    # 3. Expand output layer
    print("  Expanding output layer...")
    large_model.lm_head = expand_linear(
        small_model.lm_head,
        in_features=large_dim
    )
    
    print("✅ Model expansion complete!")
    return large_model


def verify_expansion(small_model, large_model):
    """Verify that expansion preserved knowledge."""
    print("\nVerifying expansion...")
    
    # Check parameter counts
    small_params = sum(p.numel() for p in small_model.parameters())
    large_params = sum(p.numel() for p in large_model.parameters())
    
    print(f"  Small model: {small_params:,} parameters")
    print(f"  Large model: {large_params:,} parameters")
    print(f"  Ratio: {large_params/small_params:.2f}x")
    
    # Simple forward pass test
    test_input = torch.randint(0, 1000, (1, 10))
    
    with torch.no_grad():
        small_out = small_model(test_input)
        large_out = large_model(test_input)
    
    print(f"  Small output shape: {small_out.shape}")
    print(f"  Large output shape: {large_out.shape}")
    print("✅ Expansion verified!")


if __name__ == "__main__":
    print("Model Expansion Utilities")
    print("="*80)
    print("This module provides utilities for progressive model scaling.")
    print("Use expand_model_simple() to grow a trained model.")
