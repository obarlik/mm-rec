#!/usr/bin/env python3
"""
MM-Rec Inference Demo
Loads a trained JAX model (.msgpack) and runs an interactive chat loop.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

try:
    import jax
    import jax.numpy as jnp
    from flax import serialization
    import tiktoken
    from mm_rec_jax.model.mm_rec import MMRecModel
except ImportError as e:
    print(f"❌ Error importing dependencies: {e}")
    print("Please ensure you are in the project root and have installed requirements.")
    print("Usage: python client/inference_demos.py --model path/to/model.msgpack")
    sys.exit(1)

def load_model(model_path, config_path=None):
    """Load model and parameters."""
    print(f"📂 Loading model from {model_path}...")
    print(f"   Device: {jax.devices()[0]}")
    
    # Load Config properly
    if config_path:
        with open(config_path) as f:
            file_config = json.load(f)
            # Merge with defaults but prefer file
            # Ensure MoE/UBOO/Memory flags are passed
            config = file_config
            print(f"   Config loaded: {config.keys()}")
    else:
        # Fallback defaults
        config = {
            'vocab_size': 100300,
            'model_dim': 512, # Update to match baseline
            'num_layers': 6,
            'num_heads': 8,
            'use_moe': False,
            'use_uboo': False,
            'short_mem_len': 512,
            'long_mem_len': 512
        }

    # Initialize Model
    model = MMRecModel(
        vocab_size=config.get('vocab_size', 100300),
        model_dim=config.get('model_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        max_seq_len=config.get('max_length', 2048),
        use_moe=config.get('use_moe', False),
        use_uboo=config.get('use_uboo', False),
        short_mem_len=config.get('short_mem_len', 512),
        long_mem_len=config.get('long_mem_len', 512)
    )
    
    # Init dummy state
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_mem = model.initialize_state(1)
    
    variables = model.init(rng, dummy_input, dummy_mem)
    params = variables['params']
    
    # Load Weights
    print(f"📂 Loading weights from {model_path}...")
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
        
    print("✅ Model loaded.")
    return model, params, dummy_mem

def generate(model, params, memory_state, input_ids, max_new_tokens=50, temperature=0.7):
    """Simple greedy/sampling generation with state persistence."""
    
    current_ids = input_ids
    
    # JIT-compile the single step application for speed
    @jax.jit
    def step_fn(params, x, mem):
        logits, new_mem, _ = model.apply(
            {'params': params}, 
            x, 
            mem, 
            training=False
        )
        return logits, new_mem

    generated = []
    
    print("   Thinking...", end="", flush=True)
    
    # 1. Prefill (Process history/input)
    # Update memory with input context
    logits, memory_state = step_fn(params, current_ids, memory_state)
    
    # Get last token's logits
    next_token_logits = logits[0, -1, :] 
    next_token = jnp.argmax(next_token_logits) # Greedy for now
    generated.append(int(next_token))
    
    # 2. Autoregressive Loop
    cur_token = jnp.array([[next_token]], dtype=jnp.int32)
    
    for _ in range(max_new_tokens):
        logits, memory_state = step_fn(params, cur_token, memory_state)
        
        next_token_logits = logits[0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        
        token_id = int(next_token)
        if token_id == 100257: # EOT usually
             break
             
        generated.append(token_id)
        cur_token = jnp.array([[token_id]], dtype=jnp.int32)
        
    print("\r", end="")
    return generated, memory_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .msgpack model file")
    parser.add_argument("--config", type=str, required=True, help="Path to training config.json")
    parser.add_argument("--session", type=str, default="default_user", help="Session ID for memory persistence")
    args = parser.parse_args()
    
    # Setup
    model, params, _ = load_model(args.model, args.config)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Session / Memory Management
    session_path = f"sessions/{args.session}.msgpack"
    os.makedirs("sessions", exist_ok=True)
    
    # Load valid template state (batch size 1)
    template_state = model.initialize_state(1)
    
    if os.path.exists(session_path):
        print(f"🧠 Loading existing memory for session: {args.session}")
        from mm_rec_jax.core.memory_state import MemoryState
        memory_state = MemoryState.load(session_path, template_state)
    else:
        print(f"✨ Creating new memory for session: {args.session}")
        memory_state = template_state

    print("\n💬 Chat Ready! (Ctrl+C to exit, 'save' to force save)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                # Auto save on exit
                print("💾 Saving memory...")
                memory_state.save(session_path)
                break
            
            if user_input.lower() == 'save':
                memory_state.save(session_path)
                print("💾 Memory Saved.")
                continue
                
            # Tokenize
            input_ids = tokenizer.encode(user_input)
            input_tensor = jnp.array([input_ids], dtype=jnp.int32)
            
            start = time.time()
            # Pass current memory_state, get updated memory_state back
            out_tokens, memory_state = generate(model, params, memory_state, input_tensor)
            dt = time.time() - start
            
            # Decode
            response = tokenizer.decode(out_tokens)
            print(f"Bot: {response}")
            print(f"    (Speed: {len(out_tokens)/dt:.1f} tok/s)")
            print("-" * 50)
            
        except KeyboardInterrupt:
            # Save on Interrupt too
            print("\n💾 Saving memory...")
            try:
                memory_state.save(session_path)
            except:
                pass
            print("Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
