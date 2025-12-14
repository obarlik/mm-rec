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
    
    # default config (must match training!)
    config = {
        'vocab_size': 100300,
        'model_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
    }
    
    # Allow overriding via config file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
            config.update(file_config)
            
    print(f"   Config: {config}")

    # Initialize Model
    model = MMRecModel(
        vocab_size=config['vocab_size'],
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    )
    
    # Init dummy state to get shape
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_mem = model.initialize_state(1)
    
    variables = model.init(rng, dummy_input, dummy_mem)
    params = variables['params']
    
    # Load Weights
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
        
    print("✅ Model loaded successfully.")
    return model, params, dummy_mem

def generate(model, params, memory_state, input_ids, max_new_tokens=50, temperature=0.7):
    """Simple greedy/sampling generation."""
    
    current_ids = input_ids
    
    # JIT-compile the single step application for speed
    # We pass training=False
    @jax.jit
    def step_fn(params, x, mem):
        logits, new_mem = model.apply(
            {'params': params}, 
            x, 
            mem, 
            training=False
        )
        return logits, new_mem

    generated = []
    
    print("   Thinking...", end="", flush=True)
    
    # 1. Prefill (Process history)
    # We run the whole history through to update memory state
    # Ideally we'd use the efficient scan, but for inference we just run it.
    # MMRec is RNN-like, so we consume context to update hidden state.
    
    # Note: MMRec returns sequence of outputs. We care about the LAST logits for next token prediction.
    # And we care about the FINAL memory state for the next step.
    
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
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .msgpack model file")
    parser.add_argument("--config", type=str, help="Path to training config.json")
    args = parser.parse_args()
    
    # Setup
    model, params, memory_state = load_model(args.model, args.config)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    print("\n💬 Chat Ready! (Ctrl+C to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Tokenize
            # cl100k format: messages -> tokens
            # For simplicity in demo, we feed raw text tokens
            input_ids = tokenizer.encode(user_input)
            input_tensor = jnp.array([input_ids], dtype=jnp.int32)
            
            # Generate
            # Note: We reset memory_state for each new unrelated query in this simple demo
            # Or we could keep it for multi-turn if we manage it securely.
            # Let's keep it fresh for now to avoid context pollution from previous random chats.
            fresh_mem = model.initialize_state(1)
            
            start = time.time()
            out_tokens = generate(model, params, fresh_mem, input_tensor)
            dt = time.time() - start
            
            # Decode
            response = tokenizer.decode(out_tokens)
            print(f"Bot: {response}")
            print(f"    (Speed: {len(out_tokens)/dt:.1f} tok/s)")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
