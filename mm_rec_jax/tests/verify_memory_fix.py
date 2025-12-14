
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from mm_rec_jax.model.mm_rec import MMRecModel
from mm_rec_jax.core.memory_state import MemoryState

def verify_memory_update():
    print("🧪 Starting Memory Update Verification...")
    
    # 1. Initialize Model & State
    config = {
        'model_dim': 128,
        'num_layers': 1,
        'num_heads': 4,
        'vocab_size': 1000,
        'short_mem_len': 128,
        'long_mem_len': 128,  # Small for easy testing
    }
    
    model = MMRecModel(**config)
    rng = jax.random.PRNGKey(0)
    
    # Init state
    dummy_input = jnp.zeros((1, 32), dtype=jnp.int32)
    state = model.initialize_state(batch_size=1)
    
    # Pass state to init as well
    variables = model.init(rng, dummy_input, state)
    
    print(f"Initial Long-Term Usage Mean: {jnp.mean(state.long_term.usage)}")
    
    # 2. Run Forward Pass (Simulation)
    # interacting with the model requires a valid train state or direct apply
    # We'll use apply directly
    
    print("🏃 Running Forward Pass...")
    x = jax.random.randint(rng, (1, 32), 0, 1000)
    
    # Forward pass which SHOULD update memory
    # Function returns (logits, new_state, aux_loss)
    # No 'mutable' needed as state is passed/returned functionally
    logits, new_state, aux_loss = model.apply(variables, x, state)
    
    # updated_mem_state = new_state # No dict unpacking needed
    
    logits, final_state, _ = model.apply(variables, x, state)
    
    print(f"Final Long-Term Usage Mean: {jnp.mean(final_state.long_term.usage)}")
    
    # 3. Validation
    # If update_long was called, 'usage' values should have changed (incremented or reset via LRU)
    # Initial usage is typically 0 or 1.
    # If implementation uses LRU, accessed slots get usage=1.0, others decay?
    # Or usage counts up?
    
    # Let's look at difference
    diff = jnp.sum(jnp.abs(final_state.long_term.usage - state.long_term.usage))
    print(f"Total Usage Change: {diff}")
    
    if diff > 1e-6:
        print("✅ SUCCESS: Long-Term Memory updated!")
    else:
        print("❌ FAILURE: Long-Term Memory did NOT update (Usage unchanged)")

if __name__ == "__main__":
    verify_memory_update()
