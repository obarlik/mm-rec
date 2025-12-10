#!/usr/bin/env python3
"""
C++ Optimizasyonlarının Performans Benchmark'ı

Test edilen optimizasyonlar:
1. Associative Scan (Blelloch Parallel vs Sequential)
2. Core Recurrence (C++ Fused vs PyTorch)
3. MDI (C++ SIMD vs PyTorch)
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

try:
    import mm_rec_blocks_cpu
    CPP_BLOCKS_AVAILABLE = True
except ImportError:
    print("⚠️  mm_rec_blocks_cpu bulunamadı!")
    CPP_BLOCKS_AVAILABLE = False

# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_function(func, *args, warmup=3, iterations=10, **kwargs):
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
# 1. Associative Scan Benchmark
# ============================================================================

def benchmark_associative_scan():
    """Benchmark Associative Scan: C++ vs PyTorch Sequential"""
    print("\n" + "="*70)
    print("1️⃣  ASSOCIATIVE SCAN BENCHMARK")
    print("="*70)
    
    # Test configurations
    configs = [
        {"batch": 2, "heads": 4, "seq_len": 128, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 512, "dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 2048, "dim": 64},
    ]
    
    results = []
    
    for config in configs:
        batch, heads, seq_len, dim = config["batch"], config["heads"], config["seq_len"], config["dim"]
        print(f"\n📊 Test: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}")
        
        # Create input
        gamma = torch.rand(batch, heads, seq_len, dim, dtype=torch.float32)
        gamma = torch.clamp(gamma, 0.01, 0.99)  # Valid decay range
        
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
        
        pytorch_time, pytorch_result = benchmark_function(pytorch_sequential, gamma)
        
        # C++ Optimized (Blelloch Parallel)
        if CPP_SCAN_AVAILABLE:
            cpp_time, cpp_result = benchmark_function(
                mm_rec_scan_cpu.associative_scan_exponential_cpu, gamma
            )
            
            # Verify correctness
            max_diff = torch.max(torch.abs(pytorch_result - cpp_result)).item()
            mean_diff = torch.mean(torch.abs(pytorch_result - cpp_result)).item()
            
            speedup = pytorch_time / cpp_time
            
            print(f"  PyTorch Sequential: {pytorch_time*1000:.2f} ms")
            print(f"  C++ Blelloch:       {cpp_time*1000:.2f} ms")
            print(f"  ⚡ Hızlanma:        {speedup:.2f}x")
            print(f"  ✅ Doğruluk:         max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            
            results.append({
                "config": config,
                "pytorch_time": pytorch_time,
                "cpp_time": cpp_time,
                "speedup": speedup,
                "max_diff": max_diff
            })
        else:
            print(f"  PyTorch Sequential: {pytorch_time*1000:.2f} ms")
            print(f"  ❌ C++ extension bulunamadı!")
    
    return results

# ============================================================================
# 2. Core Recurrence Benchmark
# ============================================================================

def benchmark_core_recurrence():
    """Benchmark Core Recurrence: C++ Fused vs PyTorch"""
    print("\n" + "="*70)
    print("2️⃣  CORE RECURRENCE BENCHMARK")
    print("="*70)
    
    if not CPP_BLOCKS_AVAILABLE:
        print("❌ mm_rec_blocks_cpu bulunamadı!")
        return []
    
    # Test configurations
    configs = [
        {"batch": 2, "seq_len": 128, "hidden_dim": 256},
        {"batch": 2, "seq_len": 512, "hidden_dim": 512},
        {"batch": 2, "seq_len": 2048, "hidden_dim": 1024},
    ]
    
    results = []
    
    for config in configs:
        batch, seq_len, hidden_dim = config["batch"], config["seq_len"], config["hidden_dim"]
        print(f"\n📊 Test: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        
        # Create inputs
        z_t = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)
        h_prev = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)
        W_g = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32)
        gamma = torch.rand(batch, seq_len, hidden_dim, dtype=torch.float32) * 0.5 + 0.5
        
        # PyTorch Reference
        def pytorch_recurrence(z_t, h_prev, W_g, gamma):
            # g = W_g @ h_prev
            g = torch.matmul(h_prev, W_g.t())
            # σ(g)
            gate = torch.sigmoid(g)
            # h_t = z_t ⊙ σ(g) + γ ⊙ h_prev
            h_t = z_t * gate + gamma * h_prev
            return h_t
        
        pytorch_time, pytorch_result = benchmark_function(
            pytorch_recurrence, z_t, h_prev, W_g, gamma
        )
        
        # C++ Fused
        cpp_time, cpp_result = benchmark_function(
            mm_rec_blocks_cpu.core_recurrence_fused, z_t, h_prev, W_g, gamma
        )
        
        # Verify correctness
        max_diff = torch.max(torch.abs(pytorch_result - cpp_result)).item()
        mean_diff = torch.mean(torch.abs(pytorch_result - cpp_result)).item()
        
        speedup = pytorch_time / cpp_time
        
        print(f"  PyTorch:            {pytorch_time*1000:.2f} ms")
        print(f"  C++ Fused:          {cpp_time*1000:.2f} ms")
        print(f"  ⚡ Hızlanma:        {speedup:.2f}x")
        print(f"  ✅ Doğruluk:         max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        results.append({
            "config": config,
            "pytorch_time": pytorch_time,
            "cpp_time": cpp_time,
            "speedup": speedup,
            "max_diff": max_diff
        })
    
    return results

# ============================================================================
# 3. MDI Benchmark
# ============================================================================

def benchmark_mdi():
    """Benchmark MDI: C++ SIMD vs PyTorch"""
    print("\n" + "="*70)
    print("3️⃣  MDI BENCHMARK")
    print("="*70)
    
    if not CPP_BLOCKS_AVAILABLE:
        print("❌ mm_rec_blocks_cpu bulunamadı!")
        return []
    
    # Test configurations
    configs = [
        {"batch": 2, "seq_len": 128, "model_dim": 256},
        {"batch": 2, "seq_len": 512, "model_dim": 512},
        {"batch": 2, "seq_len": 2048, "model_dim": 1024},
    ]
    
    results = []
    
    for config in configs:
        batch, seq_len, model_dim = config["batch"], config["seq_len"], config["model_dim"]
        print(f"\n📊 Test: batch={batch}, seq_len={seq_len}, model_dim={model_dim}")
        
        # Create inputs
        h_new = torch.randn(batch, seq_len, model_dim, dtype=torch.float32)
        h_old = torch.randn(batch, seq_len, model_dim, dtype=torch.float32)
        gamma = torch.rand(batch, seq_len, model_dim, dtype=torch.float32) * 0.5
        gate = torch.sigmoid(torch.randn(batch, seq_len, model_dim, dtype=torch.float32))
        
        # PyTorch Reference
        def pytorch_mdi(h_new, h_old, gamma, gate):
            # h_updated = gate ⊙ h_new + (1 - gate) ⊙ h_old + γ ⊙ h_old
            return gate * h_new + (1 - gate) * h_old + gamma * h_old
        
        pytorch_time, pytorch_result = benchmark_function(
            pytorch_mdi, h_new, h_old, gamma, gate
        )
        
        # C++ SIMD
        cpp_time, cpp_result = benchmark_function(
            mm_rec_blocks_cpu.mdi_update_fused, h_new, h_old, gamma, gate
        )
        
        # Verify correctness
        max_diff = torch.max(torch.abs(pytorch_result - cpp_result)).item()
        mean_diff = torch.mean(torch.abs(pytorch_result - cpp_result)).item()
        
        speedup = pytorch_time / cpp_time
        
        print(f"  PyTorch:            {pytorch_time*1000:.2f} ms")
        print(f"  C++ SIMD:           {cpp_time*1000:.2f} ms")
        print(f"  ⚡ Hızlanma:        {speedup:.2f}x")
        print(f"  ✅ Doğruluk:         max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        results.append({
            "config": config,
            "pytorch_time": pytorch_time,
            "cpp_time": cpp_time,
            "speedup": speedup,
            "max_diff": max_diff
        })
    
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    print("🚀 C++ Optimizasyonları Benchmark")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"mm_rec_scan_cpu: {'✅' if CPP_SCAN_AVAILABLE else '❌'}")
    print(f"mm_rec_blocks_cpu: {'✅' if CPP_BLOCKS_AVAILABLE else '❌'}")
    
    all_results = {}
    
    # 1. Associative Scan
    scan_results = benchmark_associative_scan()
    all_results["associative_scan"] = scan_results
    
    # 2. Core Recurrence
    recurrence_results = benchmark_core_recurrence()
    all_results["core_recurrence"] = recurrence_results
    
    # 3. MDI
    mdi_results = benchmark_mdi()
    all_results["mdi"] = mdi_results
    
    # Summary
    print("\n" + "="*70)
    print("📊 ÖZET")
    print("="*70)
    
    if scan_results:
        avg_speedup = sum(r["speedup"] for r in scan_results) / len(scan_results)
        print(f"\n1. Associative Scan:")
        print(f"   Ortalama Hızlanma: {avg_speedup:.2f}x")
        for r in scan_results:
            print(f"   - seq_len={r['config']['seq_len']}: {r['speedup']:.2f}x")
    
    if recurrence_results:
        avg_speedup = sum(r["speedup"] for r in recurrence_results) / len(recurrence_results)
        print(f"\n2. Core Recurrence:")
        print(f"   Ortalama Hızlanma: {avg_speedup:.2f}x")
        for r in recurrence_results:
            print(f"   - seq_len={r['config']['seq_len']}: {r['speedup']:.2f}x")
    
    if mdi_results:
        avg_speedup = sum(r["speedup"] for r in mdi_results) / len(mdi_results)
        print(f"\n3. MDI:")
        print(f"   Ortalama Hızlanma: {avg_speedup:.2f}x")
        for r in mdi_results:
            print(f"   - seq_len={r['config']['seq_len']}: {r['speedup']:.2f}x")
    
    print("\n" + "="*70)
    print("✅ Benchmark tamamlandı!")
    print("="*70)
    
    return all_results

if __name__ == "__main__":
    main()
