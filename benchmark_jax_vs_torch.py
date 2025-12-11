
import time
import torch
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.benchmark import Timer

def torch_associative_scan(x):
    """
    PyTorch implementation of associative scan (cumulative prod).
    For benchmarking, we use cumprod which is O(N) but optimized in C++.
    """
    # x: [batch, length, dim]
    return torch.cumprod(x, dim=1)

def jax_associative_scan_fn(x):
    """
    JAX implementation using jax.lax.associative_scan.
    This compiles to a logarithmic depth parallel scan.
    """
    # JAX associative_scan works on leading dimension by default or via map
    # Here we treat the sequence as the scanned axis.
    # binary operator: multiplication
    def mul(a, b): return a * b
    return jax.lax.associative_scan(mul, x, axis=1)

# Compile JAX function (JIT)
jax_associative_scan_jit = jax.jit(jax_associative_scan_fn)

def benchmark():
    print("🚀 Starting JAX vs PyTorch CPU Benchmark...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"JAX Version: {jax.__version__}")
    
    # Configuration
    BATCH_SIZE = 4
    SEQ_LEN = 32768  # 32K context
    HIDDEN_DIM = 256
    
    print(f"\n📊 Configuration: Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, Dim={HIDDEN_DIM}")
    print(f"Total Elements: {BATCH_SIZE * SEQ_LEN * HIDDEN_DIM / 1e6:.2f} Million")
    
    # 1. PyTorch Setup
    x_torch = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32)
    # Warmup
    _ = torch_associative_scan(x_torch)
    
    # 2. JAX Setup
    x_jax = jnp.array(x_torch.numpy())
    # Warmup (trigger JIT compilation)
    _ = jax_associative_scan_jit(x_jax).block_until_ready()
    print("✅ Warmup complete. Starting measurements...")
    
    # 3. Measure PyTorch
    start_time = time.time()
    for _ in range(10):
        _ = torch_associative_scan(x_torch)
    end_time = time.time()
    torch_avg = (end_time - start_time) / 10
    print(f"🔹 PyTorch Average Time: {torch_avg*1000:.2f} ms")
    
    # 4. Measure JAX
    start_time = time.time()
    for _ in range(10):
        res = jax_associative_scan_jit(x_jax)
        res.block_until_ready() # Important for async dispatch
    end_time = time.time()
    jax_avg = (end_time - start_time) / 10
    print(f"🔹 JAX (JIT) Average Time: {jax_avg*1000:.2f} ms")
    
    # Comparison
    speedup = torch_avg / jax_avg
    print(f"\n🏆 Result: JAX is {speedup:.2f}x faster than PyTorch on CPU")
    
    # Correctness Check
    torch_res = torch_associative_scan(x_torch).numpy()
    jax_res = np.array(jax_associative_scan_jit(x_jax))
    
    # Note: Small numerical differences expected due to order of operations
    # (Sequential sum vs Tree sum)
    diff = np.abs(torch_res - jax_res).mean()
    print(f"🔍 Mean Difference (correctness check): {diff:.6e}")
    
    if speedup > 2.0:
        print("\n💡 CONCLUSION: JAX provides significant speedup on CPU.")
    else:
        print("\n💡 CONCLUSION: JAX speedup is marginal or negative simple cumsum.")
        print("   Note: associative_scan is O(N) work but O(log N) span.")
        print("   PyTorch cumprod is sequential O(N). Parallel scan benefits come at very long sequences.")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"❌ Benchmark Failed: {e}")

