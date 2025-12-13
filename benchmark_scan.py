
import torch
import time
from mm_rec.core.associative_scan_torch import associative_scan_exponential_torch

def benchmark_scan():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    device = torch.device('cuda')
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Parameters matching production
    batch_size = 32
    heads = 4
    seq_len = 512
    dim = 256 # d_head?
    
    # Gamma (decay)
    gamma = torch.rand(batch_size, heads, seq_len, dim, device=device)
    
    # Warmup
    print("Warmup...")
    for _ in range(10):
        _ = associative_scan_exponential_torch(gamma)
    torch.cuda.synchronize()
    
    # Benchmark
    steps = 100
    print(f"Benchmarking {steps} steps (B={batch_size}, L={seq_len})...")
    start = time.time()
    for _ in range(steps):
        _ = associative_scan_exponential_torch(gamma)
    torch.cuda.synchronize()
    end = time.time()
    
    duration = end - start
    fps = steps / duration
    print(f"⏱️  Duration: {duration:.4f}s")
    print(f"🚀 Speed: {fps:.2f} calls/sec")
    print(f"   (Equiv to {fps * batch_size:.2f} items/sec if batched)")

if __name__ == "__main__":
    benchmark_scan()
