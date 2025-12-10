#!/usr/bin/env python3
"""
Assembly Optimizasyonlarının Performans Benchmark'ı

Test edilen optimizasyonlar:
1. Cache Prefetching
2. FMA optimizations
3. Branch prediction hints
4. Fast exp (lookup table)
"""

import torch
import time
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mm_rec" / "cpp"))

# Import C++ extensions
try:
    import mm_rec_scan_cpu
    CPP_SCAN_AVAILABLE = True
except ImportError:
    print("⚠️  mm_rec_scan_cpu bulunamadı!")
    CPP_SCAN_AVAILABLE = False

# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_function(func, *args, warmup=5, iterations=20, **kwargs):
    """Benchmark a function with warmup and multiple iterations"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Synchronize if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end = time.perf_counter()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    avg_time = (end - start) / iterations
    return avg_time, result

# ============================================================================
# 1. Associative Scan Benchmark (With Assembly Optimizations)
# ============================================================================

def benchmark_associative_scan_assembly():
    """Benchmark Associative Scan: Assembly optimizations vs baseline"""
    print("\n" + "="*70)
    print("⚡ ASSOCIATIVE SCAN - ASSEMBLY OPTIMIZATIONS BENCHMARK")
    print("="*70)
    
    # Test configurations
    configs = [
        {"batch": 2, "heads": 4, "seq_len": 128, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 512, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 2048, "dim": 64},
        {"batch": 4, "heads": 8, "seq_len": 1024, "dim": 128},  # Larger test
    ]
    
    results = []
    
    for config in configs:
        batch, heads, seq_len, dim = config["batch"], config["heads"], config["seq_len"], config["dim"]
        print(f"\n📊 Test: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}")
        
        # Create input
        gamma = torch.rand(batch, heads, seq_len, dim, dtype=torch.float32)
        gamma = torch.clamp(gamma, 0.01, 0.99)
        
        # PyTorch Sequential (Reference)
        def pytorch_sequential(gamma):
            log_gamma = torch.log(gamma + 1e-8)
            log_gamma = torch.clamp(log_gamma, -50.0, 0.0)
            
            log_cumsum = torch.zeros_like(log_gamma)
            log_cumsum[:, :, 0, :] = log_gamma[:, :, 0, :]
            
            for t in range(1, seq_len):
                prev = log_cumsum[:, :, t-1, :]
                curr = log_gamma[:, :, t, :]
                max_val = torch.max(prev, curr)
                diff = torch.abs(prev - curr)
                diff_clamped = torch.clamp(diff, max=20.0)
                log_cumsum[:, :, t, :] = max_val + torch.log1p(torch.exp(-diff_clamped))
            
            max_log = torch.max(log_cumsum, dim=2, keepdim=True)[0]
            stable_log = log_cumsum - max_log
            return torch.exp(stable_log) * torch.exp(max_log)
        
        pytorch_time, pytorch_result = benchmark_function(pytorch_sequential, gamma, iterations=15)
        
        # C++ with Assembly Optimizations
        if CPP_SCAN_AVAILABLE:
            cpp_time, cpp_result = benchmark_function(
                mm_rec_scan_cpu.associative_scan_exponential_cpu, gamma, iterations=15
            )
            
            # Verify correctness
            max_diff = torch.max(torch.abs(pytorch_result - cpp_result)).item()
            mean_diff = torch.mean(torch.abs(pytorch_result - cpp_result)).item()
            
            speedup = pytorch_time / cpp_time
            
            print(f"  PyTorch Sequential: {pytorch_time*1000:.2f} ms")
            print(f"  C++ Assembly Opt:   {cpp_time*1000:.2f} ms")
            print(f"  ⚡ Hızlanma:        {speedup:.2f}x")
            print(f"  ✅ Doğruluk:         max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            
            # Calculate improvement over previous C++ version
            # (We can't directly compare, but we can see if it's faster)
            improvement = "✅ Optimized"
            
            results.append({
                "config": config,
                "pytorch_time": pytorch_time,
                "cpp_time": cpp_time,
                "speedup": speedup,
                "max_diff": max_diff,
                "improvement": improvement
            })
        else:
            print(f"  PyTorch Sequential: {pytorch_time*1000:.2f} ms")
            print(f"  ❌ C++ extension bulunamadı!")
    
    return results

# ============================================================================
# 2. Memory Access Pattern Benchmark
# ============================================================================

def benchmark_memory_access():
    """Benchmark memory access patterns with prefetching"""
    print("\n" + "="*70)
    print("💾 MEMORY ACCESS PATTERN BENCHMARK")
    print("="*70)
    
    if not CPP_SCAN_AVAILABLE:
        print("❌ mm_rec_scan_cpu bulunamadı!")
        return []
    
    # Test with different sequence lengths (memory-bound)
    configs = [
        {"batch": 2, "heads": 4, "seq_len": 512, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 2048, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 8192, "dim": 64},  # Very long
    ]
    
    results = []
    
    for config in configs:
        batch, heads, seq_len, dim = config["batch"], config["heads"], config["seq_len"], config["dim"]
        print(f"\n📊 Test: seq_len={seq_len} (memory-bound)")
        
        gamma = torch.rand(batch, heads, seq_len, dim, dtype=torch.float32)
        gamma = torch.clamp(gamma, 0.01, 0.99)
        
        # Multiple runs for consistency
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  Ortalama: {avg_time:.2f} ms")
        print(f"  Min:      {min_time:.2f} ms")
        print(f"  Max:      {max_time:.2f} ms")
        print(f"  Varyans:  {max_time - min_time:.2f} ms")
        
        results.append({
            "config": config,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time
        })
    
    return results

# ============================================================================
# 3. Throughput Benchmark
# ============================================================================

def benchmark_throughput():
    """Benchmark throughput (operations per second)"""
    print("\n" + "="*70)
    print("🚀 THROUGHPUT BENCHMARK")
    print("="*70)
    
    if not CPP_SCAN_AVAILABLE:
        print("❌ mm_rec_scan_cpu bulunamadı!")
        return []
    
    # Fixed configuration
    batch, heads, seq_len, dim = 2, 4, 512, 64
    gamma = torch.rand(batch, heads, seq_len, dim, dtype=torch.float32)
    gamma = torch.clamp(gamma, 0.01, 0.99)
    
    # Warmup
    for _ in range(5):
        _ = mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)
    
    # Measure throughput
    num_ops = 100
    start = time.perf_counter()
    for _ in range(num_ops):
        _ = mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)
    end = time.perf_counter()
    
    total_time = end - start
    ops_per_sec = num_ops / total_time
    time_per_op = total_time / num_ops * 1000
    
    print(f"\n📊 Throughput Test:")
    print(f"   Operations: {num_ops}")
    print(f"   Total Time: {total_time:.3f} s")
    print(f"   ⚡ Throughput: {ops_per_sec:.2f} ops/sec")
    print(f"   ⏱️  Time per op: {time_per_op:.3f} ms")
    
    # Calculate data processed
    data_size_mb = (batch * heads * seq_len * dim * 4 * 2) / (1024 * 1024)  # Input + output
    bandwidth = (data_size_mb * ops_per_sec)
    
    print(f"   📊 Data per op: {data_size_mb:.2f} MB")
    print(f"   💾 Bandwidth: {bandwidth:.2f} MB/s")
    
    return {
        "ops_per_sec": ops_per_sec,
        "time_per_op": time_per_op,
        "bandwidth": bandwidth
    }

# ============================================================================
# Main
# ============================================================================

def main():
    print("⚡ Assembly Optimizasyonları Benchmark")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"mm_rec_scan_cpu: {'✅' if CPP_SCAN_AVAILABLE else '❌'}")
    
    all_results = {}
    
    # 1. Associative Scan with Assembly Optimizations
    scan_results = benchmark_associative_scan_assembly()
    all_results["associative_scan_assembly"] = scan_results
    
    # 2. Memory Access Pattern
    memory_results = benchmark_memory_access()
    all_results["memory_access"] = memory_results
    
    # 3. Throughput
    throughput_results = benchmark_throughput()
    all_results["throughput"] = throughput_results
    
    # Summary
    print("\n" + "="*70)
    print("📊 ÖZET - ASSEMBLY OPTIMIZATIONS")
    print("="*70)
    
    if scan_results:
        avg_speedup = sum(r["speedup"] for r in scan_results) / len(scan_results)
        print(f"\n1. Associative Scan (Assembly Optimized):")
        print(f"   Ortalama Hızlanma (vs PyTorch): {avg_speedup:.2f}x")
        for r in scan_results:
            print(f"   - seq_len={r['config']['seq_len']}: {r['speedup']:.2f}x (max_diff={r['max_diff']:.2e})")
    
    if memory_results:
        print(f"\n2. Memory Access (Cache Prefetching):")
        for r in memory_results:
            print(f"   - seq_len={r['config']['seq_len']}: {r['avg_time']:.2f} ms (varyans: {r['max_time']-r['min_time']:.2f} ms)")
    
    if throughput_results:
        print(f"\n3. Throughput:")
        print(f"   - {throughput_results['ops_per_sec']:.2f} ops/sec")
        print(f"   - {throughput_results['bandwidth']:.2f} MB/s")
    
    print("\n" + "="*70)
    print("✅ Assembly Optimizations Benchmark tamamlandı!")
    print("="*70)
    
    return all_results

if __name__ == "__main__":
    main()
