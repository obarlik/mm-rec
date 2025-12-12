
"""
Benchmark: Dense vs Collision-Based Sparse Training
Measures the speedup of 'Collision Detection' style computation using JAX.
"""

import time
import jax
import jax.numpy as jnp
from jax import random
from router import init_router
from sparse_block import init_block

def benchmark():
    print("🚀 Starting Collision Training Benchmark...")
    
    # Configuration
    # Reduced batch size to prevent OOM on CPU (weight gathering duplicates weights per sample)
    BATCH_SIZE = 16
    SEQ_LEN = 32 
    TOTAL_ITEMS = BATCH_SIZE * SEQ_LEN
    
    INPUT_DIM = 1024
    NUM_CLUSTERS = 64     # Concept: The 3D space has 64 distinct clusters
    CLUSTER_DIM = 256     # Each cluster has neurons equivalent to this dim
    
    TOP_K = 4             # COLLISION: Each input only collides with 4 clusters
    SPARSITY = TOP_K / NUM_CLUSTERS
    
    print(f"📊 Config: {NUM_CLUSTERS} Clusters, Activating Top-{TOP_K}")
    print(f"📉 Sparsity: Only {SPARSITY*100:.1f}% of network is active per token")
    
    key = random.PRNGKey(42)
    k1, k2, k3 = random.split(key, 3)
    
    # Initialize Models
    router = init_router(k1, INPUT_DIM, NUM_CLUSTERS, TOP_K)
    block = init_block(k2, NUM_CLUSTERS, INPUT_DIM, CLUSTER_DIM)
    
    # Dummy Input
    x = random.normal(k3, (TOTAL_ITEMS, INPUT_DIM))
    
    # 1. Benchmark DENSE (Baseline)
    # -----------------------------
    print("\n🐢 Running DENSE Baseline (Update All)...")
    
    @jax.jit
    def dense_step(x_in):
        # In dense mode, we don't route, we just compute everything
        # (Router not strictly needed for computation, but needed for fairness if integrated)
        return block.forward_dense(x_in)
    
    # Warmup
    _ = dense_step(x[:32]).block_until_ready()
    
    start = time.time()
    for _ in range(10):
        res = dense_step(x)
        res.block_until_ready()
    end = time.time()
    dense_time = (end - start) / 10
    print(f"⏱️  Dense Time: {dense_time*1000:.2f} ms")
    
    
    # 2. Benchmark SPARSE (COLLISION)
    # -------------------------------
    print("\n⚡️ Running SPARSE Collision (Update Collided)...")
    
    @jax.jit
    def sparse_step(x_in):
        # 1. Detect Collisions (Route)
        indices, gates = router.route(x_in)
        # 2. Compute only Collisions
        return block.forward_sparse(x_in, indices, gates)
        
    # Warmup
    _ = sparse_step(x[:32]).block_until_ready()
    
    start = time.time()
    for _ in range(10):
        res = sparse_step(x)
        res.block_until_ready()
    end = time.time()
    sparse_time = (end - start) / 10
    print(f"⏱️  Sparse Time: {sparse_time*1000:.2f} ms")
    
    
    # Results
    speedup = dense_time / sparse_time
    print(f"\n🏆 Speedup: {speedup:.2f}x faster")
    print(f"💡 Theoretical Max Speedup: {1/SPARSITY:.2f}x")
    print(f"   (Difference due to router overhead and gather/scatter cost)")

    if speedup > 3.0:
        print("\n✅ SUCCESS: Collision-based training is viable/highly efficient.")
    else:
        print("\n⚠️ NOTE: Speedup is modest. Check memory access patterns.")

if __name__ == "__main__":
    benchmark()
