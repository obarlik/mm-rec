import sys
import os
import time
import argparse
import os

# Prevent JAX from hogging all GPU memory (allows sharing)
import subprocess

def get_smart_memory_fraction(reserve_gb=4.0):
    """
    Calculates the optimal JAX memory fraction based on currently free GPU memory.
    Reserves 'reserve_gb' for system/other processes.
    """
    try:
        # Get Memory Info: [Free, Total] in MiB from nvidia-smi
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        # Parse first GPU
        free_mib, total_mib = map(int, result.strip().split('\n')[0].split(', '))
        used_before = total_mib - free_mib
        
        print(f"🧠 Smart Memory: Total={total_mib}MiB, Free={free_mib}MiB")
        print(f"   (Pre-existing Usage: {used_before}MiB by System/Other processes)")
        # Strategy: "Safe Minimum"
        # 4GB was too tight for Compilation/Autotuning.
        # We target 8 GB, which handles the 2GB chunks safely.
        target_mib = 8192 # 8 GB fixed
        
        # Ensure we don't exceed free memory
        if target_mib > (free_mib - 500):
            print(f"⚠️  Warning: 4GB requested but only {free_mib}MiB free. Using available.")
            target_mib = max(1024, free_mib - 500)
        
        fraction = target_mib / total_mib
        
        print(f"🧠 Smart Memory: Free={free_mib}MiB, Total={total_mib}MiB")
        print(f"   Targeting Minimal {target_mib}MiB ({fraction:.2%}) for JAX efficiency.")
        return f"{fraction:.4f}"
        
    except Exception as e:
        print(f"⚠️  Could not detect GPU memory: {e}")
        print("   Fallback to default 30%")
        return ".30"

# Configure JAX Memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = get_smart_memory_fraction(reserve_gb=2.0)

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader
import numpy as np

import json
import tiktoken
from dataclasses import dataclass

# Ensure we can import mm_rec (for Dataset)
sys.path.append(os.getcwd())
try:
    from mm_rec.data.dataset import SFTDataset, SFTDataCollator
    from mm_rec.data.chat_format import ChatMessage
except ImportError:
    print("Warning: Could not import SFTDataset. Ensure you are in project root.")
    # Dummy if needed
    pass

from mm_rec_jax.model.mm_rec import MMRecModel

@dataclass
class SFTConfig:
    max_length: int = 512
    only_predict_assistant: bool = False
    ignore_index: int = -100

def load_real_dataset(data_path, batch_size=32):
    print(f"📂 Loading dataset from {data_path}...")
    
    # 1. Load JSONL
    conversations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Parse messages to ChatMessage objects
                msgs = [ChatMessage(role=m['role'], content=m['content']) for m in data['messages']]
                conversations.append(msgs)
    
    print(f"   Found {len(conversations)} conversations.")
    
    # 2. Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # 3. Config
    config = SFTConfig(max_length=512)
    
    # 4. Dataset
    dataset = SFTDataset(conversations, tokenizer, config)
    
    # 5. Collator
    collator = SFTDataCollator(tokenizer=tokenizer)
    
    # 6. DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    return loader

def create_train_state(rng, config):
    model = MMRecModel(
        vocab_size=config['vocab_size'],
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    )
    
    # Initialize parameters
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    dummy_state = model.initialize_state(1)
    
    params = model.init(rng, dummy_input, dummy_state)['params']
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config['learning_rate'])
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    ), model

def train_step(state, batch, memory_state, rng):
    def loss_fn(params):
        # Forward
        logits, new_mem_state = state.apply_fn(
            {'params': params}, 
            batch, 
            memory_state, 
            training=True, 
            rngs={'dropout': rng}
        )
        
        # Shift logits and labels
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        
        # Cross Entropy
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        return loss, new_mem_state
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_mem_state), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Check Health Metrics for NaN prediction
    grad_norm = optax.global_norm(grads)
    # Check max value in Memory (proxy for hidden state saturation)
    # We check Short Term Memory Key magnitude
    state_max = jnp.max(jnp.abs(new_mem_state.short_term.k))
    
    metrics = {
        'grad_norm': grad_norm,
        'state_max': state_max
    }
    
    return state, loss, new_mem_state, metrics

# Manually JIT compilation to avoid decorator issues with arguments
train_step = jax.jit(train_step, donate_argnums=(0, 2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1_gpu.json')
    args = parser.parse_args()
    
    print("🚀 Initializing JAX Training...")
    print(f"   JAX Devices: {jax.devices()}")
    
    print(f"   JAX Devices: {jax.devices()}")
    
    # Setup Data (Real)
    data_path = 'data/chat_data_real.jsonl'
    if not os.path.exists(data_path):
        # Fallback for testing if file missing
        print(f"⚠️  Data file {data_path} not found. Using dummy path 'data/chat_data.jsonl' if exists.")
        data_path = 'data/chat_data.jsonl'
        
    batch_size = 32
    train_loader = load_real_dataset(data_path, batch_size=batch_size)
    num_batches = len(train_loader)
    
    # Setup Model (Config for Benchmark - MEDIUM)
    # Testing speed: Batch 32, Dim 128
    config = {
        'vocab_size': 100300, # cl100k_base has ~100k tokens
        'model_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'learning_rate': 1e-4
    }
    
    # Setup Model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    state, model = create_train_state(init_rng, config)
    
    # Memory State (Batched)
    # We init one and broadcast/vmap it?
    # MMRecModel.initialize_state gives single item.
    # We need batch of states.
    # For now, let's assume we pass single state and vmap inside? 
    # Or just vmap the apply/scan.
    # To keep it simple: Functional state is passed in.
    
    # Let's create a Batched State manually or via vmap
    single_mem_state = model.initialize_state(1)
    # PyTree leaf broadcast
    # Use jax.device_put to put on GPU
    
    # Actually, simpler: just let JAX handle it inside 'scan' if mapped.
    # But 'scan' runs PER SEQUENCE. So we need vmap over batch.
    # Our MMRecBlock code did not allow vmap yet (it takes (B, L, D)).
    # So we don't need vmap, we process batch directly.
    # But our MemoryState needs to be batched [Batch, Slots, Dim].
    
    # Init state with batch dim
    short_term = single_mem_state.short_term
    long_term = single_mem_state.long_term
    
    def expand(x): return jnp.repeat(x[None, ...], batch_size, axis=0)
    
    batched_mem_state = single_mem_state.replace(
        short_term=short_term.replace(
            k=expand(short_term.k), 
            v=expand(short_term.v), 
            age=expand(short_term.age),
            idx=jnp.repeat(short_term.idx[None, ...], batch_size, axis=0) # Broadcast scalar idx to [Batch]
        ),
        long_term=long_term.replace(
            k=expand(long_term.k), 
            v=expand(long_term.v), 
            usage=expand(long_term.usage),
            idx=jnp.repeat(long_term.idx[None, ...], batch_size, axis=0)
        )
    )
    
    print("🔥 Starting Training Loop...")
    
    # Warmup
    print("   Compiling...")
    start = time.time()
    rng, step_rng = jax.random.split(rng)
    
    # Get first batch for warmup/shape inference
    first_batch = next(iter(train_loader))
    first_input = jnp.array(first_batch['input_ids'].numpy())
    
    state, loss, batched_mem_state, _ = train_step(state, first_input, batched_mem_state, step_rng)
    print(f"   Compilation done in {time.time() - start:.2f}s")
    
    # Loop
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        rng, step_rng = jax.random.split(rng)
        
        # Convert PyTorch tensor to JAX array (Zero copy where possible)
        batch_jax = jnp.array(batch['input_ids'].numpy())
        
        state, loss, batched_mem_state, metrics = train_step(state, batch_jax, batched_mem_state, step_rng)
        
        # Periodic Monitoring
        if i % 10 == 0:
            # Check for NaN / Instability Risk
            grad_norm = metrics['grad_norm']
            state_max = metrics['state_max']
            
            # Simple "Predictive" Warning
            warning_msg = ""
            if state_max > 90.0:
                warning_msg += " ⚠️ SATURATION RISK (State > 90)"
            if grad_norm > 10.0:
                 warning_msg += " ⚠️ EXPLOSION RISK (Grad > 10)"
            
            if i % 10 == 0: # Print every 10 steps for visibility with small data
                elapsed = time.time() - t0
                avg_speed = (i + 1) / (elapsed + 1e-6)
                
                # Get VRAM Usage via nvidia-smi (Linux)
                try:
                    vram_mb = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read().strip()
                except:
                    vram_mb = "N/A"
                    
                print(f"Step {i}/{num_batches}: Loss {loss:.4f} | Speed: {avg_speed:.2f} it/s | VRAM: {vram_mb} MiB | GNorm: {grad_norm:.2f} | MaxState: {state_max:.2f}{warning_msg}")

if __name__ == '__main__':
    main()
