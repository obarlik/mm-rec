
"""
Benchmark: LSH Router vs Learned Router
Measures the overhead reduction of using Hashing instead of Top-K Sort.
"""

import time
import jax
import jax.numpy as jnp
from jax import random
from router import init_router as init_learned_router
from lsh_router import init_lsh_router

def benchmark():
    print("🚀 Starting Router Benchmark (LSH vs Learned)...")
    
    # Configuration
    BATCH_SIZE = 1024
    SEQ_LEN = 128
    TOTAL_ITEMS = BATCH_SIZE * SEQ_LEN
    
    INPUT_DIM = 1024
    NUM_CLUSTERS = 64
    TOP_K = 1 # LSH typically routes to 1 bucket (canonical)
    
    print(f"📊 Config: {TOTAL_ITEMS} items, {NUM_CLUSTERS} Clusters, Dim {INPUT_DIM}")
    
    key = random.PRNGKey(42)
    k1, k2, k3 = random.split(key, 3)
    
    # Initialize Routers
    learned_router = init_learned_router(k1, INPUT_DIM, NUM_CLUSTERS, top_k=TOP_K)
    lsh_router = init_lsh_router(k2, INPUT_DIM, NUM_CLUSTERS)
    
    # Dummy Input
    x = random.normal(k3, (TOTAL_ITEMS, INPUT_DIM))
    
    # 1. Benchmark LEARNED (O(Dim*Clusters) + Sort)
    # ---------------------------------------------
    print("\n🧠 Benchmarking LEARNED Router (Top-K)...")
    
    @jax.jit
    def learned_step(x_in):
        return learned_router.route(x_in)
        
    start = time.time()
    for _ in range(20):
        res = learned_step(x)
        res[0].block_until_ready()
    end = time.time()
    learned_time = (end - start) / 20
    print(f"⏱️  Learned Time: {learned_time*1000:.3f} ms")
    
    
    # 2. Benchmark LSH (O(Dim*log Clusters) + Hash)
    # ---------------------------------------------
    print("\n#️⃣  Benchmarking LSH Router (Hash)...")
    
    @jax.jit
    def lsh_step(x_in):
        return lsh_router.route(x_in)
        
    _ = lsh_step(x[:32])[0].block_until_ready() # Warmup
    
    start = time.time()
    for _ in range(20):
        res = lsh_step(x)
        res[0].block_until_ready()
    end = time.time()
    lsh_time = (end - start) / 20
    print(f"⏱️  LSH Time:     {lsh_time*1000:.3f} ms")
    
    # Results
    speedup = learned_time / lsh_time
    print(f"\n🏆 Speedup: {speedup:.2f}x faster routing")
    
    if speedup > 2.0:
        print("✅ SUCCESS: LSH significantly reduces routing overhead.")
    else:
        print("⚠️ NOTE: Speedup is marginal. Check bitwise op overhead.")

if __name__ == "__main__":
    benchmark()
