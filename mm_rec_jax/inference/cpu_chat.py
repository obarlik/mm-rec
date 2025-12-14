
import os
import sys
import argparse
import shutil
import jax
import jax.numpy as jnp
import msgpack
from flax import serialization
from pathlib import Path
import time

# Force CPU to avoid interfering with Training GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Add project root
sys.path.append(os.getcwd())

from mm_rec_jax.model.mm_rec import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer

WORKSPACE_DIR = Path("workspace")

def get_latest_checkpoint(job_id):
    # Find {job_id}_ckpt_epoch_{N}.msgpack
    # Filter out .saved.msgpack to avoid confusion? Or include them?
    # User wants to test "latest".
    files = list(WORKSPACE_DIR.glob(f"{job_id}_ckpt_epoch_*.msgpack"))
    if not files:
        return None
        
    def get_epoch(p):
        try:
            return int(p.name.split('_epoch_')[1].replace('.msgpack', ''))
        except:
            return -1
            
    files.sort(key=get_epoch, reverse=True)
    return files[0]

def lock_checkpoint(ckpt_path):
    """
    Copy checkpoint to a safe name so it isn't deleted by rotation.
    Appends .saved to the name.
    """
    # Target name: same but with .saved inserted before .msgpack
    # or prefix "saved_"
    
    # Current: 5f4412d6_ckpt_epoch_7.msgpack
    # Safe:    5f4412d6_ckpt_epoch_7.saved.msgpack
    # The clean up script regex: (.*)_ckpt_epoch_(\d+)\.msgpack
    # If we change extension to .saved.msgpack, regex won't match (end anchor).
    # Perfect.
    
    new_path = ckpt_path.with_suffix(".saved.msgpack")
    
    if new_path.exists():
        print(f"🔒 Checkpoint already saved as: {new_path.name}")
        return new_path
        
    print(f"💾 Preserving checkpoint (renaming logic)...")
    try:
        shutil.copy2(ckpt_path, new_path)
        print(f"✅ Locked! Saved as {new_path.name} (won't be deleted)")
        return new_path
    except Exception as e:
        print(f"⚠️ Failed to lock checkpoint: {e}")
        return ckpt_path

def chat(checkpoint_path):
    print(f"🚀 Initializing CPU Inference Engine...")
    print(f"   Model: {checkpoint_path}")
    
    # 1. Load Checkpoint
    with open(checkpoint_path, "rb") as f:
        ckpt_data = f.read()
    unpacked = msgpack.unpackb(ckpt_data, raw=False)
    
    config = unpacked.get('config')
    if not config:
        print("❌ Config not found in checkpoint.")
        return

    # 2. Init Model
    tokenizer = get_tokenizer()
    
    model = MMRecModel(
        vocab_size=config['vocab_size'],
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        max_seq_len=config.get('max_length', 512),
        use_uboo=config.get('use_uboo', False),
        use_moe=config.get('use_moe', False)
    )
    
    # 3. Init Weights
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_state = model.initialize_state(1)
    variables = model.init(rng, dummy_input, init_state, training=False)
    
    params = serialization.from_state_dict(variables['params'], unpacked['model']['params'])
    
    # 4. Compiled Step (CPU)
    print("⚙️  Compiling JAX (CPU)...")
    
    @jax.jit
    def predict_step(params, current_ids, state):
        logits, new_state, _ = model.apply(
            {'params': params},
            current_ids,
            state,
            training=False
        )
        return logits, new_state

    print("💬 Ready! (Type 'exit' to quit)")
    
    # Chat Loop
    current_state = init_state
    
    while True:
        try:
            text = input("\nYou: ")
            if text.lower() in ['exit', 'quit']: break
            
            ids = tokenizer.encode(text)
            input_tensor = jnp.array([ids], dtype=jnp.int32)
            
            # Update state with input
            logits, current_state = predict_step(params, input_tensor, current_state)
            
            # Predict
            print("MM-Rec: ", end="", flush=True)
            
            # Simple Greedy Generation
            next_id = jnp.argmax(logits[0, -1, :])
            
            for _ in range(100):
                token = int(next_id)
                word = tokenizer.decode([token])
                print(word, end="", flush=True)
                
                # Check for stop? For now simple run
                
                next_in = jnp.array([[token]], dtype=jnp.int32)
                logits, current_state = predict_step(params, next_in, current_state)
                next_id = jnp.argmax(logits[0, -1, :])
                
            print("\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    
    ckpt = get_latest_checkpoint(args.job_id)
    if ckpt:
        print(f"🔎 Found latest: {ckpt.name}")
        safe_ckpt = lock_checkpoint(ckpt)
        chat(safe_ckpt)
    else:
        print("❌ No checkpoint found.")
