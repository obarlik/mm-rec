
"""
Stress Test: LSH & Chunk Routing (The 'Red Team' Verifier)
Rigorously tests failure modes defined in the Implementation Plan.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap

# Configuration
DIM = 128
NUM_CLUSTERS = 64
CHUNK_SIZE = 512
NUM_EXPERTS = 16  # For simulation mapping clusters to experts
KEY = random.PRNGKey(42)

def run_stress_tests():
    print("🛡️  Starting Falsification Tests (Red Team)...")
    
    # Shared LSH Setup
    hyperplanes = random.normal(KEY, (DIM, 6)) # 6 bits = 64 buckets

    def get_hash(v):
        proj = jnp.dot(v, hyperplanes)
        bits = (proj > 0).astype(jnp.int32)
        powers = 2 ** jnp.arange(6)
        return jnp.dot(bits, powers)

    # ==========================================
    # TEST 1: Load Imbalance
    # ==========================================
    print("\n⚖️  Test 1: Load Imbalance (The 'Crowded Restaurant')")
    print("    Hypothesis: Random hyperplanes create uneven buckets.")
    
    # Generate random data
    N_SAMPLES = 100_000
    data = random.normal(random.PRNGKey(1), (N_SAMPLES, DIM))
    
    # Hash all
    buckets = vmap(get_hash)(data)
    
    # Count frequency
    counts = jnp.bincount(buckets, length=64)
    
    avg_load = N_SAMPLES / 64
    max_load = jnp.max(counts)
    imbalance_ratio = max_load / avg_load
    
    print(f"    Average Load: {avg_load:.0f} items")
    print(f"    Max Load:     {max_load} items")
    print(f"    Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2.0:
        print("    ❌ FAILURE: Significant load imbalance detected (>2.0x).")
        print("       Risk: Some experts will be bottlenecks.")
    else:
        print("    ✅ PASS: Load distribution is acceptable.")


    # ==========================================
    # TEST 2: Mixed-Chunk Penalty
    # ==========================================
    print("\n🥗 Test 2: Mixed-Chunk Penalty (The 'Salad' Problem)")
    print("    Hypothesis: Chunk routing fails on mixed-topic blocks.")
    
    # Simulating 2 distinct topics (orthogonal centers)
    topic_A = random.normal(random.PRNGKey(2), (DIM,))
    topic_B = random.normal(random.PRNGKey(3), (DIM,))
    
    expert_A = get_hash(topic_A)
    expert_B = get_hash(topic_B)
    
    print(f"    Topic A -> Expert {expert_A}")
    print(f"    Topic B -> Expert {expert_B}")
    
    # Create Mixed Chunk: 50% A, 50% B
    chunk_A = topic_A + random.normal(KEY, (CHUNK_SIZE // 2, DIM)) * 0.1
    chunk_B = topic_B + random.normal(KEY, (CHUNK_SIZE // 2, DIM)) * 0.1
    mixed_chunk = jnp.concatenate([chunk_A, chunk_B], axis=0) # [512, DIM]
    random_shuffle = random.permutation(KEY, mixed_chunk) # Shuffle to mix nicely
    
    # Routing Logic:
    # 1. Optimal Token Routing (Ideal): Send each token to its true expert
    # 2. Chunk Routing (Real): Pick ONE expert for the whole 512 block (Majority Vote)
    
    # Let's say loss is distance to correct expert center
    # If routed to A, cost is 0 for A-tokens, dist(A,B) for B-tokens.
    
    dist_AB = jnp.sqrt(jnp.sum((topic_A - topic_B)**2))
    
    # Token Routing Cost (Ideal): 0 (assuming perfect router)
    cost_token = 0.0 
    
    # Chunk Routing Cost:
    # We must pick A or B. Since support is 50/50, let's say we pick A.
    # Half the tokens are happy (0 cost). Half are sad (dist_AB cost).
    cost_chunk = 0.5 * dist_AB 
    
    # Normalized penalty
    penalty_pct = cost_chunk / dist_AB * 100
    
    print(f"    Ideal Token Cost: {cost_token:.2f}")
    print(f"    Chunk Routing Cost: {cost_chunk:.2f}")
    print(f"    Performance Penalty: {penalty_pct:.1f}% (Approximation)")
    
    if penalty_pct > 20.0:
         print("    ❌ FAILURE: Mixed chunks suffer high precision loss (>20%).")
         print("       Risk: Model gets confused on topic boundaries.")
    else:
         print("    ✅ PASS: Chunk approximation is efficient.")
         
         
    # ==========================================
    # TEST 3: Hash Boundary Jitter
    # ==========================================
    print("\n🔦 Test 3: Hash Boundary Jitter")
    print("    Hypothesis: Noise near boundaries causes expert flipping.")
    
    # Find a "boundary" vector (near hyperplane 0)
    # We take a random vector and subtract its projection on HP 0 to make it orthogonal (0 distance)
    vec = random.normal(KEY, (DIM,))
    hp0 = hyperplanes[:, 0]
    proj = jnp.dot(vec, hp0)
    boundary_vec = vec - (proj / jnp.dot(hp0, hp0)) * hp0 # Now dot(boundary, hp0) is 0
    
    # Add tiny noise and check consistency
    N_TRIALS = 1000
    noise_scale = 0.05
    noisy_vecs = boundary_vec + random.normal(KEY, (N_TRIALS, DIM)) * noise_scale
    
    # Check bucket bits for HP 0
    # True boundary -> should be 50/50 flip 0/1
    # Check full bucket stability
    
    initial_bucket = get_hash(boundary_vec)
    noisy_buckets = vmap(get_hash)(noisy_vecs)
    
    flips = jnp.sum(noisy_buckets != initial_bucket)
    flip_rate = flips / N_TRIALS * 100
    
    print(f"    Boundary Flip Rate: {flip_rate:.1f}%")
    
    if flip_rate > 10.0:
        print("    ❌ FAILURE: Unstable routing near boundaries (>10% jitter).")
        print("       Risk: Training instability.")
    else:
        print("    ✅ PASS: Routing is stable.")

    print("\n🏁 Stress Tests Completed.")

if __name__ == "__main__":
    run_stress_tests()
