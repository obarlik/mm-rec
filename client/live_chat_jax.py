#!/usr/bin/env python3
"""
Live Chat with Training Model (JAX)
1. Finds latest checkpoint on Phoenix.
2. Downloads it via SCP.
3. Loads model in JAX.
4. Interactive Chat.
"""
import os
import sys
import subprocess
import jax
import jax.numpy as jnp
from flax import serialization
import msgpack
import argparse
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from mm_rec_jax.model.mm_rec import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer

SERVER_HOST = "phoenix"
REMOTE_WS = "/mnt/c/Users/Onur/mm-rec-training/workspace"

def get_latest_remote_checkpoint(job_id):
    print(f"🔍 Connecting to {SERVER_HOST} to list workspace files...")
    print("   (Please enter password if prompted)")
    try:
        # List all files in workspace to avoid glob expansion issues
        cmd = [
            "ssh", SERVER_HOST, 
            f"ls -1 {REMOTE_WS}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ SSH Command failed (Code {result.returncode})")
            print(f"   Stderr: {result.stderr.strip()}")
            return None
            
        all_files = result.stdout.strip().split('\n')
        # Filter for this job's checkpoints
        checkpoints = []
        for f in all_files:
            f = f.strip()
            if f.startswith(f"{job_id}_ckpt_epoch_") and f.endswith(".msgpack"):
                checkpoints.append(f)
                
        if not checkpoints:
            print(f"❌ No checkpoints found for job {job_id} in {REMOTE_WS}")
            print(f"   Found {len(all_files)} total files.")
            return None
            
        # Sort by epoch number
        # Format: {jobid}_ckpt_epoch_{N}.msgpack
        def get_epoch(name):
            try:
                # split by '_epoch_' and take last part, remove .msgpack
                return int(name.split('_epoch_')[1].replace('.msgpack', ''))
            except:
                return -1
                
        checkpoints.sort(key=get_epoch, reverse=True)
        
        latest = checkpoints[0]
        print(f"✅ Found latest: {latest} (Epoch {get_epoch(latest)})")
        
        # Return full relative path for SCP
        return f"{REMOTE_WS}/{latest}"
        
    except Exception as e:
        print(f"❌ Error searching remote files: {e}")
        return None

def download_checkpoint(remote_path, local_path):
    if os.path.exists(local_path):
        # Check if remote is newer? For now just overwrite if requested or assume current is ok?
        # User wants live test, so we should probably re-download if it's a new test.
        # But let's ask or just do it.
        pass
        
    print(f"📥 Downloading {remote_path} -> {local_path}...")
    try:
        subprocess.check_call(["scp", f"{SERVER_HOST}:{remote_path}", local_path])
        print("✅ Download complete.")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def chat(checkpoint_path, config_path=None):
    print("🧠 Initializing JAX Model...")
    
    # 1. Load Checkpoint Data first to get Config if embedded, or use Default
    with open(checkpoint_path, "rb") as f:
        ckpt_data = f.read()
    unpacked = msgpack.unpackb(ckpt_data, raw=False)
    
    # We need config to init model. 
    # Current msgpack structure in train_server_jax.py:
    # {'model': ..., 'opt': ..., 'config': ..., 'step': ...}
    # Let's hope 'config' is there.
    
    if 'config' in unpacked:
        config = unpacked['config']
    else:
        # Fallback to local config file if provided
        if not config_path:
            print("⚠️ Config not found in checkpoint. Please provide --config.")
            return
        import json
        with open(config_path) as f:
             config = json.load(f)

    # 2. Init Model
    tokenizer = get_tokenizer()
    vocab_size = config.get('vocab_size', 100300)
    
    model = MMRecModel(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        max_seq_len=config.get('max_length', 512),
        use_uboo=config.get('use_uboo', False),
        use_moe=config.get('use_moe', False)
    )
    
    # 3. Restore Weights
    print("⚖️  Restoring Weights...")
    # Create dummy input for init
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    
    # Init state (batched=1)
    init_state = model.initialize_state(1)
    
    variables = model.init(rng, dummy_input, init_state, training=False)
    
    # Replace params with loaded ones
    # unpacked['model']['params'] matches structure?
    # Flax serialization.from_bytes or just dictionary loading?
    # Msgpack gives dicts. calling serialization.from_state_dict equivalent
    
    params = serialization.from_state_dict(variables['params'], unpacked['model']['params'])
    
    # 4. Inference Function (JIT)
    print("🚀 Compiling Inference Step...")
    
    @jax.jit
    def generate_step(params, current_ids, state):
        # Forward pass
        # current_ids: [1, 1]
        logits, new_state, _ = model.apply(
            {'params': params},
            current_ids,
            state,
            training=False
        )
        # Greedy decode
        next_token = jnp.argmax(logits[0, -1, :])
        return next_token, new_state

    # 5. Chat Loop
    print("\n💬 MM-Rec Live Chat (Type 'exit' to quit)")
    print("-" * 50)
    
    # Reset State for session
    current_state = init_state
    
    history_tokens = [] # List of ints
    
    while True:
        try:
            user_text = input("\nYou: ")
            if user_text.lower() in ['exit', 'quit']: break
            
            # Tokenize
            # OpenAI tokenizer doesn't add BOS?
            ids = tokenizer.encode(user_text)
            history_tokens.extend(ids)
            
            # Context Window Management (Simple truncate if needed)
            MAX_CTX = 512
            if len(history_tokens) > MAX_CTX:
                history_tokens = history_tokens[-MAX_CTX:]
            
            # Feed input to update state and generate
            # For Recurrent model, we can feed token by token or chunk.
            # To be efficient, feed the *new* user input to update state.
            
            input_tensor = jnp.array([ids], dtype=jnp.int32) # [1, Seq]
            
            # Run over input to warm up state
            # (In reality we want to generate AFTER this)
            
            # Optimization: We need to run the model on `input_tensor` 
            # and keep the state, but we don't care about output logits yet.
            # BUT wait, this is a recurrent model. We assume `state` captures history.
            # So we feed `ids` and get `new_state`.
            
            # Step 1: Process User Input
            # We can't use generate_step (it predicts *next* token).
            # We need a `forward_sequence` function.
            
            @jax.jit
            def process_sequence(params, seq, state):
                _, new_state, _ = model.apply({'params': params}, seq, state, training=False)
                return new_state
            
            current_state = process_sequence(params, input_tensor, current_state)
            
            # Step 2: Generate Response
            print("MM-Rec: ", end="", flush=True)
            response_tokens = []
            
            # Start generation - prompt with last user token? 
            # The model is trained to predict NEXT token.
            # So after processing user input, the model is ready to predict response?
            # Yes, if we take the logits from the *last* step of input processing.
            # But process_sequence returned just state.
            
            # Let's verify prediction logic.
            # Feed [UserTokens] -> Model updates state -> Predicts First Response Token?
            # Actually, `model(input)` returns logits for *each* position.
            # The logit at `-1` is the prediction for the token *after* the input.
            
            # Redefine process to return last token prediction
            @jax.jit
            def process_and_predict(params, seq, state):
                logits, new_state, _ = model.apply({'params': params}, seq, state, training=False)
                next_id = jnp.argmax(logits[0, -1, :])
                return next_id, new_state
            
            # We re-run process on input to get first token
            next_token_id, current_state = process_and_predict(params, input_tensor, init_state) # Ah! Re-using Init state?
            # NO! We must carry state if we want multi-turn.
            # BUT, for the *current* user input, we should feed it into the *accumulated* state?
            # Yes. 
            
            # Re-implementation for Chat:
            # 1. User Input -> Feed to Model (Update State) -> Get predicted next token
            # 2. Loop until <EOS> or limit
            
            # Wait, if we use `current_state` (accumulated), we just feed new tokens.
            # Correct.
            
            # Re-compile to be safe with shapes? JIT handles compilation cache.
            
            # Feed user input to update state and get first generation token
            next_token_id, current_state = process_and_predict(params, input_tensor, current_state)
            
            gen_count = 0
            while gen_count < 100: # Max response len
                token_val = int(next_token_id)
                word = tokenizer.decode([token_val])
                print(word, end="", flush=True)
                response_tokens.append(token_val)
                
                # Check Stop (newline or special?)
                # For chat, maybe just newline or specific token?
                # Data is 'messages' format. Usually has <|endoftext|> or special tokens.
                # Assuming standard text generation for now.
                
                # Next step
                next_input = jnp.array([[token_val]], dtype=jnp.int32)
                next_token_id, current_state = process_and_predict(params, next_input, current_state)
                gen_count += 1
                
            print("\n")
            
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--download", action="store_true", help="Force download latest")
    args = parser.parse_args()
    
    latest_ckpt_path = f"client/{args.job_id}_latest.msgpack"
    
    remote_file = get_latest_remote_checkpoint(args.job_id)
    if remote_file:
        if args.download or not os.path.exists(latest_ckpt_path):
            download_checkpoint(remote_file, latest_ckpt_path)
            
    if os.path.exists(latest_ckpt_path):
        chat(latest_ckpt_path)
    else:
        print("❌ Could not obtain a checkpoint.")
