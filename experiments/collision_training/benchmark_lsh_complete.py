
"""
Benchmark: Dense vs LSH-Based Sparse Training
Measures the TOTAL speedup of using LSH routing + Sparse Block vs Dense Baseline.
"""

import time
import jax
import jax.numpy as jnp
from jax import random
from lsh_router import init_lsh_router
from sparse_block import init_block

def benchmark():
    print("🚀 Starting Complete Benchmark (Dense vs LSH-Sparse)...")
    
    # Configuration
    # Using the safe batch size from previous fix
    BATCH_SIZE = 16
    SEQ_LEN = 32
    TOTAL_ITEMS = BATCH_SIZE * SEQ_LEN
    
    INPUT_DIM = 1024
    NUM_CLUSTERS = 64
    CLUSTER_DIM = 256
    
    # LSH routes to 1 bucket usually
    TOP_K = 1 
    SPARSITY = TOP_K / NUM_CLUSTERS
    
    print(f"📊 Config: {NUM_CLUSTERS} Clusters, LSH routes to single bucket (Top-1)")
    print(f"📉 Sparsity: {SPARSITY*100:.1f}% active")
    
    key = random.PRNGKey(42)
    k1, k2, k3 = random.split(key, 3)
    
    # Initialize
    lsh_router = init_lsh_router(k1, INPUT_DIM, NUM_CLUSTERS)
    block = init_block(k2, NUM_CLUSTERS, INPUT_DIM, CLUSTER_DIM)
    
    # Dummy Input
    x = random.normal(k3, (TOTAL_ITEMS, INPUT_DIM))
    
    # 1. Benchmark DENSE
    # ------------------
    print("\n🐢 Running DENSE Baseline...")
    
    @jax.jit
    def dense_step(x_in):
        return block.forward_dense(x_in)
    
    _ = dense_step(x[:SAMPLE_SIZE]) if (SAMPLE_SIZE := min(32, TOTAL_ITEMS)) else None
    
    start = time.time()
    for _ in range(20):
        res = dense_step(x)
        res.block_until_ready()
    end = time.time()
    dense_time = (end - start) / 20
    print(f"⏱️  Dense Time: {dense_time*1000:.2f} ms")
    
    
    # 2. Benchmark LSH-SPARSE
    # -----------------------
    print("\n⚡️ Running LSH-SPARSE...")
    
    @jax.jit
    def lsh_sparse_step(x_in):
        # 1. Route with LSH (Super fast)
        indices, gates = lsh_router.route(x_in)
        # 2. Compute Sparse (Only 1 cluster per token)
        return block.forward_sparse(x_in, indices, gates)
        
    _ = lsh_sparse_step(x[:SAMPLE_SIZE])
    
    start = time.time()
    for _ in range(20):
        res = lsh_sparse_step(x)
        res.block_until_ready()
    end = time.time()
    sparse_time = (end - start) / 20
    print(f"⏱️  LSH-Sparse Time: {sparse_time*1000:.2f} ms")
    
    # Results
    speedup = dense_time / sparse_time
    print(f"\n🏆 Speedup: {speedup:.2f}x faster")
    
    if speedup > 3.0:
        print("✅ SUCCESS: LSH+Sparse is significantly faster than Dense.")
    elif speedup > 1.0:
        print("⚠️ MARGINAL: LSH is faster, but overhead is still high.")
    else:
        print("❌ SLOWER: Overhead dominates. Needs larger model/GPU.")

if __name__ == "__main__":
    benchmark()
