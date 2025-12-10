"""
Model Upscaling Utilities
Küçük modelden büyük modele weight transfer ve upscaling
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from mm_rec.model import MMRecModel


def transfer_weights_smart(
    source_model: MMRecModel,
    target_model: MMRecModel,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Küçük modelden büyük modele akıllı weight transfer.
    
    Strateji:
    1. Embedding: Vocab size farkı varsa, ortak kısmı transfer et
    2. Blocks: Layer sayısı farkı varsa, mevcut layer'ları transfer et, yeni layer'ları initialize et
    3. Layer norm: Aynı dimension'ları transfer et
    4. Output head: Vocab size farkı varsa, ortak kısmı transfer et
    
    Args:
        source_model: Küçük model (kaynak)
        target_model: Büyük model (hedef)
        verbose: Detaylı log göster
    
    Returns:
        Transfer edilen parametrelerin durumu
    """
    transferred = {}
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    # 1. Embedding transfer
    if 'embedding.weight' in source_state and 'embedding.weight' in target_state:
        src_emb = source_state['embedding.weight']
        tgt_emb = target_state['embedding.weight']
        
        if src_emb.shape[0] <= tgt_emb.shape[0] and src_emb.shape[1] == tgt_emb.shape[1]:
            # Ortak vocab kısmını transfer et
            tgt_emb[:src_emb.shape[0], :] = src_emb
            target_state['embedding.weight'] = tgt_emb
            transferred['embedding'] = True
            if verbose:
                print(f"✅ Embedding transferred: {src_emb.shape[0]} -> {tgt_emb.shape[0]} vocab")
        else:
            transferred['embedding'] = False
    
    # 2. Block transfer (layer-by-layer)
    source_num_layers = len(source_model.blocks)
    target_num_layers = len(target_model.blocks)
    
    num_transferred_layers = min(source_num_layers, target_num_layers)
    
    for layer_idx in range(num_transferred_layers):
        # Block parametrelerini transfer et
        block_transferred = transfer_block_weights(
            source_model.blocks[layer_idx],
            target_model.blocks[layer_idx],
            source_state,
            target_state,
            layer_idx,
            verbose
        )
        transferred[f'block_{layer_idx}'] = block_transferred
    
    # Yeni layer'lar initialize edilmiş olarak kalır (random init)
    if target_num_layers > source_num_layers:
        if verbose:
            print(f"ℹ️  {target_num_layers - source_num_layers} yeni layer initialize edildi (random)")
    
    # 3. Final norm transfer
    if 'norm.weight' in source_state and 'norm.weight' in target_state:
        src_norm = source_state['norm.weight']
        tgt_norm = target_state['norm.weight']
        
        if src_norm.shape == tgt_norm.shape:
            target_state['norm.weight'] = src_norm
            transferred['norm'] = True
            if verbose:
                print("✅ Final norm transferred")
        else:
            transferred['norm'] = False
    
    # 4. Output head transfer
    if 'lm_head.weight' in source_state and 'lm_head.weight' in target_state:
        src_head = source_state['lm_head.weight']
        tgt_head = target_state['lm_head.weight']
        
        if src_head.shape[0] <= tgt_head.shape[0] and src_head.shape[1] == tgt_head.shape[1]:
            # Ortak vocab kısmını transfer et
            tgt_head[:src_head.shape[0], :] = src_head
            target_state['lm_head.weight'] = tgt_head
            transferred['lm_head'] = True
            if verbose:
                print(f"✅ Output head transferred: {src_head.shape[0]} -> {tgt_head.shape[0]} vocab")
        else:
            transferred['lm_head'] = False
    
    # State dict'i yükle
    target_model.load_state_dict(target_state, strict=False)
    
    return transferred


def transfer_block_weights(
    source_block: nn.Module,
    target_block: nn.Module,
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    layer_idx: int,
    verbose: bool
) -> bool:
    """
    Tek bir block'un weight'lerini transfer et.
    """
    transferred = False
    
    # Block prefix
    src_prefix = f'blocks.{layer_idx}.'
    tgt_prefix = f'blocks.{layer_idx}.'
    
    # HEM: W_fused veya W_q, W_k, W_v, W_z
    if hasattr(source_block, 'W_fused') and hasattr(target_block, 'W_fused'):
        # Fused weight transfer (boyutlar farklı olabilir)
        src_key = src_prefix + 'W_fused.weight'
        tgt_key = tgt_prefix + 'W_fused.weight'
        
        if src_key in source_state and tgt_key in target_state:
            src_w = source_state[src_key]
            tgt_w = target_state[tgt_key]
            
            # Model dimension aynıysa, ortak kısmı transfer et
            if src_w.shape[1] == tgt_w.shape[1]:  # Input dimension aynı
                min_out = min(src_w.shape[0], tgt_w.shape[0])
                tgt_w[:min_out, :] = src_w[:min_out, :]
                target_state[tgt_key] = tgt_w
                transferred = True
                if verbose:
                    print(f"  ✅ Block {layer_idx}: W_fused transferred ({min_out}/{tgt_w.shape[0]} dims)")
    
    # DPG: W_gamma_down, W_gamma_up
    if hasattr(source_block, 'W_gamma_down') and hasattr(target_block, 'W_gamma_down'):
        src_key = src_prefix + 'W_gamma_down.weight'
        tgt_key = tgt_prefix + 'W_gamma_down.weight'
        
        if src_key in source_state and tgt_key in target_state:
            src_w = source_state[src_key]
            tgt_w = target_state[tgt_key]
            
            if src_w.shape == tgt_w.shape:
                target_state[tgt_key] = src_w
                transferred = True
    
    # MDI: W_g, W_gamma, W_context
    mdi_components = ['W_g.weight', 'W_gamma.0.weight', 'W_gamma.2.weight']
    if hasattr(source_block, 'mdi') and hasattr(target_block, 'mdi'):
        for component in mdi_components:
            src_key = src_prefix + 'mdi.' + component
            tgt_key = tgt_prefix + 'mdi.' + component
            
            if src_key in source_state and tgt_key in target_state:
                src_w = source_state[src_key]
                tgt_w = target_state[tgt_key]
                
                if src_w.shape == tgt_w.shape:
                    target_state[tgt_key] = src_w
                    transferred = True
    
    # FFN: ffn.0.weight, ffn.3.weight
    ffn_components = ['ffn.0.weight', 'ffn.3.weight']
    for component in ffn_components:
        src_key = src_prefix + component
        tgt_key = tgt_prefix + component
        
        if src_key in source_state and tgt_key in target_state:
            src_w = source_state[src_key]
            tgt_w = target_state[tgt_key]
            
            # Ortak dimension'ları transfer et
            if src_w.shape[1] == tgt_w.shape[1]:  # Input dimension aynı
                min_out = min(src_w.shape[0], tgt_w.shape[0])
                tgt_w[:min_out, :] = src_w[:min_out, :]
                target_state[tgt_key] = tgt_w
                transferred = True
    
    return transferred


def upscale_model(
    source_model: MMRecModel,
    target_config,
    device: torch.device,
    verbose: bool = True
) -> MMRecModel:
    """
    Küçük modelden büyük model oluştur ve weight'leri transfer et.
    
    Args:
        source_model: Küçük model (kaynak)
        target_config: Hedef model konfigürasyonu (ModelConfig)
        device: Device
        verbose: Detaylı log
    
    Returns:
        Upscaled model (weight'ler transfer edilmiş)
    """
    if verbose:
        print(f"🔄 Upscaling model: {source_model.num_layers} layers -> {target_config.num_layers} layers")
        print(f"   Model dim: {source_model.model_dim} -> {target_config.model_dim}")
        print(f"   Vocab: {source_model.vocab_size} -> {target_config.vocab_size}")
    
    # Hedef modeli oluştur
    target_model = MMRecModel(
        vocab_size=target_config.vocab_size,
        model_dim=target_config.model_dim,
        num_layers=target_config.num_layers,
        num_heads=target_config.num_heads,
        num_memories=target_config.num_memories,
        mem_dim=target_config.mem_dim,
        ffn_dim=target_config.ffn_dim,
        max_seq_len=target_config.max_seq_len,
        dropout=target_config.dropout,
        use_hem=target_config.use_hem,
        use_dpg=target_config.use_dpg,
        use_uboo=target_config.use_uboo,
        dpg_rank=target_config.dpg_rank,
        lambda_P=target_config.lambda_P
    ).to(device)
    
    # Weight transfer
    transferred = transfer_weights_smart(source_model, target_model, verbose=verbose)
    
    if verbose:
        print(f"✅ Model upscaled: {sum(transferred.values())}/{len(transferred)} components transferred")
    
    return target_model
