
"""
Benchmark: Scaling Law Analysis (Overhead vs Scale)
Tests how Chunk-Routing overhead changes as model dimension increases up to 16384.
"""

import time
import jax
import jax.numpy as jnp
from jax import random
import gc

def benchmark_scale():
    print("🚀 Starting Scaling Benchmark (1024 -> 16384)...")
    
    # Dimensions to test
    DIMS = [1024, 2048, 4096, 8192, 16384]
    
    # Fixed Config
    NUM_CHUNKS = 16     # Fewer chunks to speed up benchmark
    CHUNK_SIZE = 512
    NUM_EXPERTS = 8     # Kept modest to avoid OOM at 16k dim (8 * 1GB = 8GB weights)
    
    results = []

    for DIM in DIMS:
        HIDDEN = DIM
        print(f"\n📏 Testing Dimension: {DIM} (Weights: {DIM}x{DIM})")
        
        # Force garbage collection between runs
        gc.collect()
        jax.clear_caches()
        
        key = random.PRNGKey(42)
        k1, k2, k3 = random.split(key, 3)
        
        # Initialize
        # Note: At 16384, this is 8 experts * 1GB = 8GB VRAM/RAM
        try:
            weights = random.normal(k1, (NUM_EXPERTS, DIM, HIDDEN)) * 0.01
            x = random.normal(k2, (NUM_CHUNKS, CHUNK_SIZE, DIM))
            indices = random.randint(k3, (NUM_CHUNKS,), 0, NUM_EXPERTS)
            
            # Dense Weight (Simulated Average)
            w_dense = jnp.mean(weights, axis=0) # [DIM, HIDDEN]
        except Exception as e:
            print(f"❌ OOM or Error at Dim {DIM}: {e}")
            break
            
        # 1. Dense Baseline
        @jax.jit
        def dense_fwd(x_in):
            x_flat = x_in.reshape(-1, DIM)
            return jnp.dot(x_flat, w_dense)
            
        # Warmup
        _ = dense_fwd(x[:1]).block_until_ready()
        
        start = time.time()
        for _ in range(10): # Reduced iterations for large dims
            _ = dense_fwd(x).block_until_ready()
        dense_time = (time.time() - start) / 10
        
        # 2. Chunk Sparse
        @jax.jit
        def chunk_sparse_fwd(x_in, expert_ids):
            def scan_fn(carry, inputs):
                chunk_x, expert_idx = inputs
                w_selected = weights[expert_idx]
                out = jnp.dot(chunk_x, w_selected)
                return carry, out
            _, outputs = jax.lax.scan(scan_fn, None, (x_in, expert_ids))
            return outputs
            
        # Warmup
        _ = chunk_sparse_fwd(x[:1], indices[:1]).block_until_ready()
        
        start = time.time()
        for _ in range(10):
            _ = chunk_sparse_fwd(x, indices).block_until_ready()
        sparse_time = (time.time() - start) / 10
        
        # Analysis
        ratio = sparse_time / dense_time
        overhead = (ratio - 1.0) * 100
        
        print(f"   Dense: {dense_time*1000:.1f}ms | Sparse: {sparse_time*1000:.1f}ms")
        print(f"   Relative Cost: {ratio:.2f}x ({overhead:+.1f}%)")
        
        results.append((DIM, ratio))
        
        # Clean up large arrays
        del weights, x, indices, w_dense
        
    print("\n📈 Final Scaling Trend:")
    print("Dimension | Relative Cost (Lower is better)")
    print("-------------------------------------------")
    for dim, ratio in results:
        print(f"{dim:<9} | {ratio:.2f}x")
        
    if results[-1][1] < results[0][1]:
        print("\n✅ HYPOTHESIS CONFIRMED: Overhead decreases as model gets larger.")
    else:
        print("\n❌ HYPOTHESIS REJECTED: Overhead remains constant or increases.")

if __name__ == "__main__":
    benchmark_scale()
