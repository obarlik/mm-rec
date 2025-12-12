
"""
Benchmark: Chunk-Level Routing (Cache-Aware Sparsity)
Measures the speedup of routing *chunks* of tokens to experts, ensuring cache locality.
"""

import time
import jax
import jax.numpy as jnp
from jax import random

def benchmark():
    print("🚀 Starting Chunk-Level Routing Benchmark...")

    # Configuration
    # We simulate a long sequence split into chunks
    NUM_CHUNKS = 64
    CHUNK_SIZE = 512  # Cache friendly size
    TOTAL_SEQ = NUM_CHUNKS * CHUNK_SIZE
    BATCH_SIZE = 1    # Focusing on sequence processing speed
    
    DIM = 1024
    HIDDEN = 1024
    NUM_EXPERTS = 16  # Number of possible experts
    
    print(f"📊 Config: Sequence Length {TOTAL_SEQ}, Chunk Size {CHUNK_SIZE}")
    print(f"   Experts: {NUM_EXPERTS}, Dim: {DIM}")
    
    key = random.PRNGKey(42)
    k1, k2, k3, k4 = random.split(key, 4)
    
    # Weights for all experts: [num_experts, dim, hidden]
    weights = random.normal(k1, (NUM_EXPERTS, DIM, HIDDEN)) * 0.02
    
    # Input: [num_chunks, chunk_size, dim] (Pre-reshaped for benchmark)
    x = random.normal(k2, (NUM_CHUNKS, CHUNK_SIZE, DIM))
    
    # Randomly assign each chunk to an expert
    # [num_chunks]
    chunk_expert_indices = random.randint(k3, (NUM_CHUNKS,), 0, NUM_EXPERTS)
    
    # 1. Benchmark Dense (Baseline)
    # -----------------------------
    # Standard large matmul: treats all weights as one large averaged weight 
    # or just runs a standard layer. To be fair, let's say Dense = 1 active large expert.
    # A standard Transformer FFN is one large matrix.
    # Expert Mix is usually larger total params, but we compare computational cost.
    # Dense cost: 1 Matmul of size [Total, Dim] * [Dim, Hidden]
    
    # Averaged weight for dense baseline simulation
    w_dense = jnp.mean(weights, axis=0)
    
    print("\n🐢 Running Dense Baseline (Standard Matmul)...")
    
    @jax.jit
    def dense_fwd(x_in):
        # x_in: [chunks, chunk_size, dim] -> [total_seq, dim]
        x_flat = x_in.reshape(-1, DIM)
        return jnp.dot(x_flat, w_dense)

    _ = dense_fwd(x).block_until_ready()
    
    start = time.time()
    for _ in range(50):
        _ = dense_fwd(x).block_until_ready()
    end = time.time()
    dense_time = (end - start) / 50
    print(f"⏱️  Dense Time: {dense_time*1000:.3f} ms")


    # 2. Benchmark Chunk-Sparse
    # -------------------------
    # We iterate over chunks, pick THE weight matrix for that chunk, and matmul.
    # This keeps W_expert in cache for 512 steps.
    
    print("\n⚡️ Running Chunk-Sparse Routing...")
    
    @jax.jit
    def chunk_sparse_fwd(x_in, expert_ids):
        # Scan over chunks
        def scan_fn(carry, inputs):
            chunk_x, expert_idx = inputs
            
            # 1. Select Weight (Gather 1 matrix)
            # This is the only "random access" - once per 512 tokens!
            w_selected = weights[expert_idx] 
            
            # 2. Compute Chunk (Dense Matmul for this chunk)
            # [512, 1024] @ [1024, 1024]
            out = jnp.dot(chunk_x, w_selected)
            
            return carry, out
            
        _, outputs = jax.lax.scan(scan_fn, None, (x_in, expert_ids))
        return outputs

    _ = chunk_sparse_fwd(x, chunk_expert_indices).block_until_ready()
    
    start = time.time()
    for _ in range(50):
        _ = chunk_sparse_fwd(x, chunk_expert_indices).block_until_ready()
    end = time.time()
    sparse_time = (end - start) / 50
    print(f"⏱️  Chunk-Sparse Time: {sparse_time*1000:.3f} ms")
    
    # Results
    speedup = dense_time / sparse_time
    print(f"\n🏆 Relative Cost: {sparse_time/dense_time*100:.1f}% of Dense")
    print(f"   (Ideally should be close to 100% or faster if total params >> active)")
    
    # Note on Speedup:
    # If standard Dense layer has size W [D, H], and our sparse experts have TOTAL size N * [D, H],
    # We are comparing executing 1 huge layer vs 1 selected small layer.
    # Actually Model Capacity of Sparse is N times larger.
    # So fair comparison is: How fast is Sparse compared to a Dense layer of SAME ACTIVE PARAMS?
    # They should be roughly equal (1.0x). If Sparse is much slower (e.g. 0.2x speedup), then overhead is high.
    # If Sparse is close to 1.0x (e.g. 0.9x), it's a WIN because we get N times more parameter capacity for "free".

    overhead = sparse_time - dense_time
    print(f"💡 Overhead per chunk: {overhead/NUM_CHUNKS*1000:.4f} ms")
    
    if sparse_time < dense_time * 1.5:
        print("✅ SUCCESS: Chunk-Routing is efficient. You get 16x parameters for almost same compute cost.")
    else:
        print("⚠️ NOTE: Slice/Scan overhead is visible.")

if __name__ == "__main__":
    benchmark()
