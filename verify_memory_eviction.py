
import jax
import jax.numpy as jnp
from mm_rec_jax.core.memory_state import MemoryState, MemoryBank

def test_eviction():
    print("🧠 Testing 'Survival of the Fittest' Memory...")
    
    # 1. Init (Short=2, Long=4)
    model_dim = 4
    state = MemoryState.create(model_dim, 2, model_dim, 4)
    
    # Fill Long Term with dummy data (A, B, C, D)
    # IDs: 1, 2, 3, 4
    init_k = jnp.stack([jnp.ones(4)*i for i in range(1, 5)]) # [[1,1..], [2,2..]...]
    init_v = init_k
    
    # Manually set long term state for test
    state = state.replace(
        long_term=state.long_term.replace(
            k=init_k, v=init_v, usage=jnp.ones(4) # Usage = 1.0
        )
    )
    
    print(f"Initial State indices: {[1, 2, 3, 4]}")
    print(f"Initial Usage: {state.long_term.usage}")
    
    # 2. Simulate Usage
    # Increase usage of slot 1 and 3 (index 0 and 2)
    # Usage -> [5.0, 1.0, 8.0, 1.0]
    new_usage = jnp.array([5.0, 1.0, 8.0, 0.5]) # D is least used (0.5)
    state = state.replace(long_term=state.long_term.replace(usage=new_usage))
    print(f"Simulated Usage: {state.long_term.usage}")
    
    # 3. Incoming Data (E, F)
    # IDs: 5, 6
    # Batch=1, N=2, Dim=4
    k_in = jnp.stack([jnp.ones(4)*5, jnp.ones(4)*6])[None, ...] # [1, 2, 4]
    v_in = k_in
    
    # We need to treat 'state' as Batched [1, 4, 4] too for vmap to work as expected
    # OR rely on the broadcasting logic inside update_long
    # The current broadcasting logic:
    # if current_k.ndim == 2: batch_size = k_new.shape[0]; broadcast...
    # If k_in is [1, 2, 4], batch_size=1.
    
    # 4. Trigger Update Long
    state = state.update_long(k_in, v_in)
    
    # Output is now Batched [1, 4, 4]
    final_k = state.long_term.k[0] 
    final_usage = state.long_term.usage[0]
    
    print("\n--- After Consolidation ---")
    print(f"Final Usage: {final_usage}")
    print(f"Final Keys (First element): {final_k[:, 0]}")
    
    # Check if 5 and 6 exist
    has_5 = jnp.any(final_k[:, 0] == 5)
    has_6 = jnp.any(final_k[:, 0] == 6)
    
    # Check if High Usage items (1 and 3) survived
    has_1 = jnp.any(final_k[:, 0] == 1)
    has_3 = jnp.any(final_k[:, 0] == 3)
    
    print(f"\nAnalysis:")
    print(f"New Memory (5) Inserted? {has_5}")
    print(f"New Memory (6) Inserted? {has_6}")
    print(f"Strong Memory (1) Survived? {has_1}")
    print(f"Strong Memory (3) Survived? {has_3}")
    
    if has_5 and has_6 and has_1 and has_3:
        print("\n✅ SUCCESS: Weak memories evicted, Strong preserved.")
    else:
        print("\n❌ FAILED: Logic error.")
        # Debug print
        print(f"Expected [1, 5, 3, 6] (order may vary)")

if __name__ == "__main__":
    test_eviction()
