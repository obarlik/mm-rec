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
    Returns: (fraction_string, free_vram_gb)
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
        
        # Return free VRAM in GB for validation
        free_gb = free_mib / 1024.0
        
        # Strategy: Use ~95% of FREE VRAM (not total)
        # Leave 5% buffer for system overhead and fragmentation
        usable_mib = max(1024, free_mib - 500)  # Reserve 500MiB for safety
        target_mib = int(usable_mib * 0.95)      # Use 95% of available
        
        fraction = target_mib / total_mib
        
        print(f"🧠 Smart Memory: Total={total_mib}MiB, Free={free_mib}MiB")
        print(f"   (Pre-existing Usage: {total_mib - free_mib}MiB by System/Other processes)")
        print(f"   Allocating {target_mib}MiB ({fraction:.2%} of total, ~95% of free) for JAX")
        return f"{fraction:.4f}", free_gb
        
    except Exception as e:
        print(f"⚠️  Could not detect GPU memory: {e}")
        print("   Fallback to default 30%")
        return ".30", 16.0  # Assume 16GB if detection fails


# Configure JAX Memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
MEMORY_FRACTION, FREE_VRAM_GB = get_smart_memory_fraction(reserve_gb=2.0)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = MEMORY_FRACTION

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader
import numpy as np

import json
import tiktoken
import glob
from flax import serialization
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

def load_dataset_memory(data_path, max_length=512):
    print(f"📂 Loading dataset from {data_path} into RAM...")
    
    # 1. Load JSONL
    conversations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                msgs = [ChatMessage(role=m['role'], content=m['content']) for m in data['messages']]
                conversations.append(msgs)
    
    print(f"   Found {len(conversations)} conversations.")
    
    # 2. Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Robust pad_id
    pad_id = 0
    if hasattr(tokenizer, 'pad_token_id'): pad_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'eot_token'): pad_id = tokenizer.eot_token
    
    # 3. Config
    config = SFTConfig(max_length=max_length)
    
    # 4. Dataset (Use SFTDataset for parsing logic)
    dataset = SFTDataset(conversations, tokenizer, config)
    
    # 5. Tokenize & Pad ALL
    print("   Tokenizing and Padding all samples...")
    all_input_ids = []
    for i in range(len(dataset)):
        item = dataset[i]
        ids = item['input_ids'] # Already truncated by Dataset
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
            
        # Manual Padding
        if len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))
        all_input_ids.append(ids)
        
    # Stack to JAX Array
    full_data = jnp.array(all_input_ids, dtype=jnp.int32)
    print(f"   ✅ Loaded {full_data.shape} into device memory.")
    return full_data

def create_train_state(rng, config):
    model = MMRecModel(
        vocab_size=config['vocab_size'],
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_uboo=config.get('use_uboo', False),
        use_moe=config.get('use_moe', False)
    )
    
    # Initialize parameters
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    dummy_state = model.initialize_state(1)
    
    params = model.init(rng, dummy_input, dummy_state)['params']
    
    # Adaptive Learning Rate Schedule
    # Warmup (first 5% of training) + Cosine Decay
    total_steps = config['num_epochs'] * (1400 // config['batch_size'])  # Approx steps
    warmup_steps = int(total_steps * config.get('warmup_fraction', 0.05))
    
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config['learning_rate'],
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=config['learning_rate'] * 0.1  # Decay to 10% of peak
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_schedule)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    ), model

def save_checkpoint(state, epoch, filename):
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(state))
    print(f"💾 Checkpoint saved: {filename}")

def restore_checkpoint(state, filename):
    with open(filename, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    print(f"♻️  Resumed from checkpoint: {filename}")
    return state

def train_step(state, batch, memory_state, rng):
    def loss_fn(params):
        # Forward
        # Forward (Returns: logits, new_state, aux_loss)
        logits, new_mem_state, aux_loss = state.apply_fn(
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
        main_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        
        # Total Loss = Main Loss + Aux Loss
        loss = main_loss + aux_loss
        
        return loss, (new_mem_state, main_loss, aux_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_mem_state, main_loss, aux_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Check Health Metrics for NaN prediction
    grad_norm = optax.global_norm(grads)
    # Check max value in Memory (proxy for hidden state saturation)
    # We check Short Term Memory Key magnitude
    state_max = jnp.max(jnp.abs(new_mem_state.short_term.k))
    
    metrics = {
        'grad_norm': grad_norm,
        'state_max': state_max,
        'main_loss': main_loss,
        'aux_loss': aux_loss
    }
    
    return state, loss, new_mem_state, metrics

# Manually JIT compilation to avoid decorator issues with arguments
train_step = jax.jit(train_step, donate_argnums=(0, 2))

def validate_and_adjust_config(config, free_vram_gb):
    """
    Validate training config against available GPU memory.
    Auto-adjust batch_size if needed to prevent OOM.
    
    Returns: (adjusted_config, was_adjusted, warning_msg)
    """
    # Estimated VRAM usage formula (rough heuristic):
    # VRAM ≈ model_params + activations + optimizer_states + data
    # For safe operation, we use empirically determined limits:
    # Batch Size 16 requires ~8-10GB VRAM (from previous tests)
    # Batch Size 8 requires ~4-6GB VRAM
    
    SAFE_BATCH_SIZES = {
        # free_vram_gb: max_safe_batch_size
        24: 16,  # RTX 4090 with 24GB can handle batch 16
        20: 16,
        16: 12,
        12: 8,
        8: 4,
        4: 2,
    }
    
    was_adjusted = False
    warning_msg = ""
    original_batch_size = config.get('batch_size', 8)
    
    # Find appropriate batch size for available VRAM
    max_safe_batch = 2  # Default minimum
    for vram_threshold in sorted(SAFE_BATCH_SIZES.keys()):
        if free_vram_gb >= vram_threshold:
            max_safe_batch = SAFE_BATCH_SIZES[vram_threshold]
    
    if original_batch_size > max_safe_batch:
        config['batch_size'] = max_safe_batch
        was_adjusted = True
        warning_msg = f"⚠️  VRAM SAFETY: Batch size reduced from {original_batch_size} to {max_safe_batch} (Free VRAM: {free_vram_gb:.1f}GB)"
        print(warning_msg)
    else:
        print(f"✅ VRAM Check Passed: Batch size {original_batch_size} is safe for {free_vram_gb:.1f}GB free VRAM")
    
    return config, was_adjusted, warning_msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1_gpu.json')
    args = parser.parse_args()
    
    print("🚀 Initializing JAX Training...", flush=True)
    print(f"   JAX Devices: {jax.devices()}", flush=True)
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help='Path to JSON config file')
    args, unknown = parser.parse_known_args()

    # Default Config
    default_config = {
        'vocab_size': 100300, # cl100k_base has ~100k tokens
        'model_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'learning_rate': 1e-4,
        'data_path': 'data/chat_data_real.jsonl',
        'batch_size': 32,
        'num_epochs': 5
    }

    # Load Config from File (Merge with Defaults)
    config = default_config.copy()
    if args.config and os.path.exists(args.config):
        print(f"📄 Loading config from {args.config}", flush=True)
        with open(args.config) as f:
            file_config = json.load(f)
            # Update defaults with file config (only for keys that exist or new keys)
            config.update(file_config)
    else:
        print("⚠️  No config file provided. Using Defaults.")

    # ==== SMART VRAM VALIDATION ====
    # Validate config against actual GPU capacity and auto-adjust if needed
    print("\n🔍 Validating Config Against GPU Capacity...", flush=True)
    config, was_adjusted, warning_msg = validate_and_adjust_config(config, FREE_VRAM_GB)
    
    if was_adjusted:
        # Save adjusted config back to file for transparency
        if args.config:
            adjusted_path = args.config.replace('.json', '_adjusted.json')
            with open(adjusted_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Saved adjusted config to: {adjusted_path}")


    # Setup Data (Real)
    data_path = config.get('data_path', 'data/chat_data_real.jsonl')
    if not os.path.exists(data_path):
        # Fallback for testing if file missing
        print(f"⚠️  Data file {data_path} not found. Using dummy fallback.")
        data_path = 'data/chat_data.jsonl'
        
    full_data = load_dataset_memory(data_path, max_length=512)
    num_samples = full_data.shape[0]
    batch_size = config.get('batch_size', 32)
    num_batches = num_samples // batch_size
    print(f"   Batch Size: {batch_size}, Batches/Epoch: {num_batches}")
    
    # Setup Model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    state, model = create_train_state(init_rng, config)
    
    # Memory State (Batched)
    batched_mem_state = model.initialize_state(config['batch_size'])
    
    # RESUME LOGIC

    # RESUME LOGIC
    # Look for existing checkpoints for this job or generic ones
    # Checkpoint format expected: "{job_id}_ckpt_epoch_{e}.msgpack" OR "ckpt_epoch_{e}.msgpack"
    start_epoch = 0
    global_step = 0
    
    # We need the JOB ID from config to be specific
    current_job_id = config.get('job_name', 'default') # Weak fallback
    # Better: Scan directory for ANY msgpack that looks like a checkpoint
    checkpoint_dir = "workspace"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.msgpack') and 'ckpt_epoch' in f]
    
    if checkpoints:
        # Sort by modification time (latest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        
        try:
            print(f"♻️  Found checkpoint: {latest_ckpt}. Attempting resume...")
            state = restore_checkpoint(state, latest_ckpt)
            
            # Extract Epoch from filename if possible
            # Format: 'd1f21a35_ckpt_epoch_1.msgpack'
            import re
            match = re.search(r'ckpt_epoch_(\d+)', latest_ckpt)
            if match:
                resumed_epoch = int(match.group(1))
                start_epoch = resumed_epoch # Continue from NEXT epoch? No, usually saved at END of epoch.
                # So if we saved Epoch 1, we start Epoch 1? Or 2?
                # Usually we save AFTER epoch finishes. So we should start at next.
                # But let's assume we start AT that epoch index if 0-indexed.
                # Let's say we saved epoch 1 (0-indexed 1 means 2nd epoch done).
                # Actually, safety first: restart the epoch or assume finished?
                # Let's rely on global_step if stored, but we don't store it in filename.
                # JAX TrainState stores 'step' internally!
                
                # Get step from loaded state
                global_step = int(state.step)
                print(f"   Resumed at Step {global_step} (Epoch ~{start_epoch})")
                
                # Update start_epoch based on global_step
                start_epoch = global_step // num_batches
                print(f"   Corrected Start Epoch to {start_epoch}")
                
        except Exception as e:
             print(f"⚠️  Resume failed: {e}. Starting from scratch.")

    
    print("🔥 Starting Training Loop...")
    
    # Warmup
    print("   Compiling...")
    start = time.time()
    rng, step_rng = jax.random.split(rng)
    
    # Get first batch for warmup/shape inference
    first_input = full_data[:batch_size]
    
    state, loss, batched_mem_state, _ = train_step(state, first_input, batched_mem_state, step_rng)
    print(f"   Compilation done in {time.time() - start:.2f}s")
    
    # Loop
    num_epochs = config.get('num_epochs', 5)
    print(f"🔥 Starting Training Loop for {num_epochs} Epochs...")
    
    t0 = time.time()
    global_step = 0
    initial_step = 0
    start_epoch = 0
    
    # Check for Checkpoints to Resume
    # Pattern: {base}_ckpt_epoch_{n}.msgpack
    # We need 'base' name
    config_path = args.config if args.config else "default_config.json"
    base_name = os.path.splitext(config_path)[0]
    if base_name.endswith("_config"): base_name = base_name.replace("_config", "")
    
    ckpt_pattern = f"{base_name}_ckpt_epoch_*.msgpack"
    ckpts = glob.glob(ckpt_pattern)
    if ckpts:
        # Sort by epoch number
        # Filename format: ..._epoch_{n}.msgpack
        try:
            ckpts.sort(key=lambda x: int(x.split('_epoch_')[1].split('.msgpack')[0]))
            latest = ckpts[-1]
            print(f"🔎 Found existing checkpoints. Resuming from {latest}...")
            state = restore_checkpoint(state, latest)
            
            # Determine start epoch
            resumed_epoch = int(latest.split('_epoch_')[1].split('.msgpack')[0])
            start_epoch = resumed_epoch
            # global_step approximation if needed, but not critical for AdamW resume (state has step)
            # Actually TrainState stores step!
            global_step = int(state.step)
            initial_step = global_step
            print(f"   Resuming at Epoch {start_epoch}, Step {global_step}")
        except Exception as e:
            print(f"⚠️ Failed to resume from checkpoint: {e}")
            
    # Early Stopping Setup
    early_stop_patience = config.get('early_stop_patience', 3)  # Stop after 3 epochs without improvement
    min_delta = config.get('min_delta', 0.02)  # Minimum improvement threshold
    best_epoch_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"   Epoch {epoch+1}/{num_epochs}")
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        
        # Shuffle Data for this Epoch
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, num_samples)
        # We must truncate to multiple of batch_size manually since we don't have drop_last logic here
        # Actually simplest is to just slice carefully
        
        epoch_data = full_data[perms]
        
        for i in range(num_batches):
            global_step += 1
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Slice batch (Zero copy in JAX ideally)
            batch_jax = epoch_data[start_idx:end_idx]
            
            state, loss, batched_mem_state, metrics = train_step(state, batch_jax, batched_mem_state, step_rng)
            
            # Accumulate epoch loss
            epoch_loss_sum += float(loss)
            epoch_step_count += 1
            
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
                
                elapsed = time.time() - t0
                # Fix: Speed should be based on steps processed IN THIS SESSION
                steps_this_session = global_step - initial_step
                avg_speed = steps_this_session / (elapsed + 1e-6)
                
                # Calculate ETA
                total_expected_steps = num_epochs * num_batches
                remaining_steps = total_expected_steps - global_step
                eta_seconds = remaining_steps / (avg_speed + 1e-6)
                eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))
                if eta_seconds > 3600:
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                
                # Get VRAM Usage via nvidia-smi (Linux)
                try:
                    vram_mb = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read().strip()
                except:
                    vram_mb = "N/A"
                    
                print(f"Epoch {epoch+1} | Step {global_step} (E:{i}/{num_batches}): Loss {loss:.4f} | Speed: {avg_speed:.2f} it/s | ETA: {eta_str} | VRAM: {vram_mb} MiB | GNorm: {grad_norm:.2f} | MaxState: {state_max:.2f}{warning_msg}", flush=True)
            
            # Periodic Step-Based Checkpoint (every 1000 steps = ~12 mins @ 1.4 it/s)
            checkpoint_interval = config.get('checkpoint_interval', 1000)
            if global_step % checkpoint_interval == 0:
                step_ckpt_path = f"{base_name}_ckpt_step_{global_step}.msgpack"
                save_checkpoint(state, epoch+1, step_ckpt_path)
                print(f"💾 Checkpoint saved at step {global_step}", flush=True)

        # Calculate Epoch Average Loss
        epoch_avg_loss = epoch_loss_sum / max(epoch_step_count, 1)
        epoch_losses.append(epoch_avg_loss)
        
        # Early Stopping Check
        improvement = best_epoch_loss - epoch_avg_loss
        
        if improvement > min_delta:
            # Significant improvement
            best_epoch_loss = epoch_avg_loss
            patience_counter = 0
            print(f"📈 Epoch {epoch+1} Avg Loss: {epoch_avg_loss:.4f} (↓ {improvement:.4f}) - Best so far!")
        else:
            patience_counter += 1
            print(f"📉 Epoch {epoch+1} Avg Loss: {epoch_avg_loss:.4f} (↓ {improvement:.4f}) - Patience: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️  Early Stopping: No improvement for {early_stop_patience} epochs (min_delta={min_delta})")
                print(f"   Best Epoch Loss: {best_epoch_loss:.4f}")
                break
        
        # Save Checkpoint at End of Epoch
        ckpt_path = f"{base_name}_ckpt_epoch_{epoch+1}.msgpack"
        save_checkpoint(state, epoch+1, ckpt_path)

    # SAVE MODEL
    print("💾 Saving Model...")
    from flax import serialization
    
    # Check config source to determine filename
    if args.config:
        # e.g. workspace/1234abcd_config.json -> workspace/1234abcd_model.msgpack
        base = os.path.splitext(args.config)[0]
        if base.endswith("_config"):
            save_path = base.replace("_config", "_model") + ".msgpack"
        else:
            save_path = base + "_model.msgpack"
    else:
        save_path = "model.msgpack"
        
    with open(save_path, "wb") as f:
        f.write(serialization.to_bytes(state.params))
    
    print(f"✅ Model saved to {save_path}")

if __name__ == '__main__':
    main()
