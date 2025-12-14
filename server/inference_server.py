
import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional, List, Dict

# Force JAX to use CPU for Inference Server to avoid VRAM conflict with Training
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn

# Project imports
sys.path.append(os.getcwd())

try:
    import jax
    import jax.numpy as jnp
    from flax import serialization
    import tiktoken
    from mm_rec_jax.model.mm_rec import MMRecModel
    from mm_rec_jax.core.memory_state import MemoryState
except ImportError as e:
    print(f"❌ Error importing JAX/Project modules: {e}")
    sys.exit(1)

app = FastAPI(title="MM-Rec Inference Server (CPU)")

# Globals
model: Optional[MMRecModel] = None
params: Optional[Dict] = None
tokenizer = None
CONFIG = {}

# Paths
SESSION_DIR = Path("sessions")
SESSION_DIR.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_new_tokens: int = 50
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    speed_tok_sec: float
    session_id: str

def load_model_and_params(model_path: str, config_path: str):
    """Load model once on startup."""
    global model, params, tokenizer, CONFIG
    
    print(f"📂 Loading Config from {config_path}...")
    with open(config_path) as f:
        CONFIG = json.load(f)
    
    print(f"📂 Loading Model from {model_path}...")
    # Initialize Model
    model = MMRecModel(
        vocab_size=CONFIG.get('vocab_size', 100300),
        model_dim=CONFIG.get('model_dim', 512),
        num_layers=CONFIG.get('num_layers', 6),
        num_heads=CONFIG.get('num_heads', 8),
        max_seq_len=CONFIG.get('max_length', 2048),
        use_moe=CONFIG.get('use_moe', False),
        use_uboo=CONFIG.get('use_uboo', False),
        short_mem_len=CONFIG.get('short_mem_len', 512),
        long_mem_len=CONFIG.get('long_mem_len', 512)
    )
    
    # Init dummy
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_mem = model.initialize_state(1)
    variables = model.init(rng, dummy_input, dummy_mem)
    params = variables['params']
    
    # Load weights
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
        
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"✅ Model Loaded on {jax.devices()[0]}")

@jax.jit
def step_fn(params, x, mem):
    """Single step inference function (JIT compiled)."""
    # Note: We need to bind 'model' via closure or pass it, but JIT functions 
    # typically shouldn't capture large objects directly if they change.
    # Since 'model' structure is static, we can likely call model.apply directly 
    # if defined globally OR use partial.
    # Typically: model.apply is pure.
    logits, new_mem, _ = model.apply(
        {'params': params}, 
        x, 
        mem, 
        training=False
    )
    return logits, new_mem

def run_inference(session_id: str, text: str, max_tokens: int):
    """Run inference with state persistence."""
    global model, params, tokenizer
    
    session_file = SESSION_DIR / f"{session_id}.msgpack"
    
    # 1. Load State
    template_state = model.initialize_state(1)
    if session_file.exists():
        memory_state = MemoryState.load(str(session_file), template_state)
    else:
        memory_state = template_state
        
    # 2. Tokenize
    input_ids = tokenizer.encode(text)
    input_tensor = jnp.array([input_ids], dtype=jnp.int32)
    
    # 3. Generate Loop
    start_t = time.time()
    
    # Prefill
    logits, memory_state = step_fn(params, input_tensor, memory_state)
    next_token = jnp.argmax(logits[0, -1, :])
    generated = [int(next_token)]
    
    cur_token = jnp.array([[next_token]], dtype=jnp.int32)
    
    for _ in range(max_tokens):
        logits, memory_state = step_fn(params, cur_token, memory_state)
        next_token = jnp.argmax(logits[0, -1, :])
        token_id = int(next_token)
        if token_id == 100257: # EOT
            break
        generated.append(token_id)
        cur_token = jnp.array([[token_id]], dtype=jnp.int32)
        
    dt = time.time() - start_t
    
    # 4. Save State
    memory_state.save(str(session_file))
    
    return tokenizer.decode(generated), len(generated)/dt

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    response_text, speed = run_inference(req.session_id, req.message, req.max_new_tokens)
    
    return ChatResponse(
        response=response_text,
        speed_tok_sec=speed,
        session_id=req.session_id
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    load_model_and_params(args.model, args.config)
    
    print("🚀 Starting Inference Server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
