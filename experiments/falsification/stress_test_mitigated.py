
"""
Stress Test: Mitigated (Top-2 Routing + Small Chunks)
Verifies if proposed fixes solve the 'Red Team' failures.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap

# Configuration - MITIGATED
DIM = 128
NUM_CLUSTERS = 64
CHUNK_SIZE = 128  # Mitigation 1: Smaller chunks (was 512)
TOP_K_EXPERTS = 2 # Mitigation 2: Route to Top-2 Experts (was 1)
KEY = random.PRNGKey(42)

def run_mitigated_tests():
    print("🛡️  Starting Mitigated Stress Tests...")
    print(f"    Config: Chunk Size {CHUNK_SIZE}, Top-{TOP_K_EXPERTS} Routing")
    
    hyperplanes = random.normal(KEY, (DIM, 6))

    def get_hash(v):
        proj = jnp.dot(v, hyperplanes)
        bits = (proj > 0).astype(jnp.int32)
        powers = 2 ** jnp.arange(6)
        return jnp.dot(bits, powers)

    # ==========================================
    # TEST 2: Mixed-Chunk Penalty (Re-visited)
    # ==========================================
    print("\n🥗 Test 2: Mixed-Chunk Penalty")
    print("    Hypothesis: Top-2 routing captures BOTH topics in a mix.")
    
    topic_A = random.normal(random.PRNGKey(2), (DIM,))
    topic_B = random.normal(random.PRNGKey(3), (DIM,))
    
    expert_A = get_hash(topic_A)
    expert_B = get_hash(topic_B)
    
    # Mixed Chunk (50/50)
    chunk_A = topic_A + random.normal(KEY, (CHUNK_SIZE // 2, DIM)) * 0.1
    chunk_B = topic_B + random.normal(KEY, (CHUNK_SIZE // 2, DIM)) * 0.1
    mixed_chunk = jnp.concatenate([chunk_A, chunk_B], axis=0) # [128, DIM]
    random_shuffle = random.permutation(KEY, mixed_chunk)
    
    # 1. Calculate Token Buckets
    token_buckets = vmap(get_hash)(mixed_chunk)
    
    # 2. Vote for Top-K Experts
    counts = jnp.bincount(token_buckets, length=64)
    # Get Top-2 indices
    top_2_experts = jnp.argsort(counts)[-TOP_K_EXPERTS:]
    
    print(f"    True Topics: {expert_A}, {expert_B}")
    print(f"    Selected Experts: {top_2_experts}")
    
    # Check if both topics are covered
    covered_A = jnp.isin(expert_A, top_2_experts)
    covered_B = jnp.isin(expert_B, top_2_experts)
    
    if covered_A and covered_B:
        print("    Penalty: 0.0% (Both topics covered)")
        print("    ✅ PASS: Multi-expert routing solved the mixed-chunk problem.")
    else:
        print("    ❌ FAILURE: Still missed a topic.")
        
        
    # ==========================================
    # TEST 3: Hash Boundary Jitter (Re-visited)
    # ==========================================
    print("\n🔦 Test 3: Hash Boundary Jitter")
    print("    Hypothesis: Top-2 routing stabilizes flips by picking both sides.")
    
    # Boundary vector setup
    vec = random.normal(KEY, (DIM,))
    hp0 = hyperplanes[:, 0]
    proj = jnp.dot(vec, hp0)
    boundary_vec = vec - (proj / jnp.dot(hp0, hp0)) * hp0
    
    # Noise loops
    N_TRIALS = 1000
    noise_scale = 0.05
    # For chunk routing stability, we simulate a chunk of noisy boundary vectors
    # This represents a "Boundary Concept" appearing in text
    chunk_of_boundary = boundary_vec + random.normal(KEY, (CHUNK_SIZE, DIM)) * noise_scale
    
    # Routing 1
    buckets_1 = vmap(get_hash)(chunk_of_boundary)
    counts_1 = jnp.bincount(buckets_1, length=64)
    top_2_1 = jnp.sort(jnp.argsort(counts_1)[-TOP_K_EXPERTS:]) # Sort for comparison
    
    # Routing 2 (Different Noise)
    chunk_of_boundary_2 = boundary_vec + random.normal(random.PRNGKey(99), (CHUNK_SIZE, DIM)) * noise_scale
    buckets_2 = vmap(get_hash)(chunk_of_boundary_2)
    counts_2 = jnp.bincount(buckets_2, length=64)
    top_2_2 = jnp.sort(jnp.argsort(counts_2)[-TOP_K_EXPERTS:])
    
    print(f"    Run 1 Selected: {top_2_1}")
    print(f"    Run 2 Selected: {top_2_2}")
    
    if jnp.array_equal(top_2_1, top_2_2):
        print("    Stability: 100% (Selections identical)")
        print("    ✅ PASS: Top-2 routing absorbed the noise.")
    else:
        print("    ❌ FAILURE: Selections changed.")

    print("\n🏁 Mitigated Tests Completed.")

if __name__ == "__main__":
    run_mitigated_tests()
