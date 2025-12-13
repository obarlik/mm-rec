import sys
import os
import time
import argparse
import os

# Prevent JAX from hogging all GPU memory (allows sharing)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Optional: Limit to 50% if needed, but false should suffice for demand-paging
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader
import numpy as np

# Ensure we can import mm_rec (for Dataset)
sys.path.append(os.getcwd())
try:
    from mm_rec.data.dataset import SFTDataset, SFTCollate
except ImportError:
    print("Warning: Could not import SFTDataset. Ensure you are in project root.")
    # Dummy if needed
    pass

from mm_rec_jax.model.mm_rec import MMRecModel

def get_collate_fn(tokenizer):
    def collate(batch):
        # reuse logic or simple stack
        # assuming batch is list of dicts
        ids = [item['input_ids'] for item in batch]
        # Pad
        max_len = max(len(x) for x in ids)
        padded = np.zeros((len(ids), max_len), dtype=np.int32)
        for i, x in enumerate(ids):
            padded[i, :len(x)] = x
        return {'input_ids': padded}
    return collate

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
    return state, loss, new_mem_state

# Manually JIT compilation to avoid decorator issues with arguments
train_step = jax.jit(train_step, donate_argnums=(0, 2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1_gpu.json')
    args = parser.parse_args()
    
    print("🚀 Initializing JAX Training...")
    print(f"   JAX Devices: {jax.devices()}")
    
    # Setup Data
    # 1. Load Dataset (PyTorch)
    # dataset = SFTDataset(...)
    # For benchmark speed, generate random data
    print("ℹ️  Generating Random Data for Benchmark (Medium)...")
    batch_size = 32
    seq_len = 512
    num_batches = 500
    
    data = [np.random.randint(0, 32000, (batch_size, seq_len)).astype(np.int32) for _ in range(num_batches)]
    
    # Setup Model (Config for Benchmark - MEDIUM)
    # Testing throughput: Batch 32, Dim 128
    config = {
        'vocab_size': 32000,
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
    state, loss, batched_mem_state = train_step(state, jnp.array(data[0]), batched_mem_state, step_rng)
    print(f"   Compilation done in {time.time() - start:.2f}s")
    
    # Loop
    t0 = time.time()
    for i in range(1, num_batches):
        rng, step_rng = jax.random.split(rng)
        batch_np = data[i]
        batch_jax = jnp.array(batch_np) # Zero copy if possible
        
        state, loss, batched_mem_state = train_step(state, batch_jax, batched_mem_state, step_rng)
        
        if i % 50 == 0:
            elapsed = time.time() - t0
            avg_speed = i / elapsed
            print(f"Step {i}: Loss {loss:.4f} | Speed: {avg_speed:.2f} it/s")

if __name__ == '__main__':
    main()
