
"""
Final Benchmark: Optimization Verification
Compares:
1. Baseline: Standard Dense FFN (Small)
2. Sparse: Sparse FFN (64 Experts, Top-2) -> Massive Capacity
3. Theoretical Huge Dense: A Dense FFN with equivalent capacity to Sparse (64x larger)

Goal: Show that Sparse achieves "Huge" capacity for "Small" cost.
"""

import torch
import torch.nn as nn
import time
from mm_rec.blocks.sparse_mm_rec_block import SparseFFN

def benchmark_integration():
    print("🚀 Starting Final Optimization Benchmark...")
    
    # Config
    B, L, D = 4, 1024, 1024
    NUM_EXPERTS = 64
    CHUNK_SIZE = 128
    
    print(f"    Input: [{B}, {L}, {D}]")
    print(f"    Experts: {NUM_EXPERTS} (Top-2 Routing)")
    
    # 1. Baseline Dense FFN (The "Small" model)
    # Params: D -> 4D -> D (Standard Transformer FFN)
    # For fair comparison with our Linear Experts (D->D), let's use Linear Dense (D->D)
    baseline_dense = nn.Linear(D, D)
    
    # 2. Sparse FFN (The "Smart" model)
    # Capacity: 64 * (D->D)
    sparse_ffn = SparseFFN(
        model_dim=D,
        chunk_size=CHUNK_SIZE,
        num_experts=NUM_EXPERTS,
        ffn_dim=D, # Linear expert
        dropout=0.0
    )
    
    # 3. Huge Dense (The "Expensive" alternative)
    # To match Sparse capacity (64 experts), we would need 64x wider layer?
    # Or just 64 parallel layers. Let's simulate the COST of running 64 dense layers.
    # We won't allocate it (OOM risk), just multiply baseline time by 64.
    
    x = torch.randn(B, L, D)
    
    # Warmup
    for _ in range(5):
        _ = baseline_dense(x)
        _ = sparse_ffn(x)
        
    # Measure Baseline
    start = time.time()
    for _ in range(50):
        _ = baseline_dense(x)
    base_time = (time.time() - start) / 50
    
    # Measure Sparse
    start = time.time()
    for _ in range(50):
        _ = sparse_ffn(x)
    sparse_time = (time.time() - start) / 50
    
    print("\n📊 Results (avg ms per forward pass):")
    print("-" * 50)
    print(f"1. Standard Dense (Small Capability):   {base_time*1000:.2f} ms")
    print(f"2. Sparse FFN    (Huge Capability):    {sparse_time*1000:.2f} ms")
    
    # Derived Metrics
    ratio = sparse_time / base_time
    huge_dense_time = base_time * NUM_EXPERTS # Theoretical cost of full activation
    savings = huge_dense_time / sparse_time
    
    print("-" * 50)
    print(f"Relative Cost: Sparse is {ratio:.1f}x slower than Small Dense.")
    print(f"Optimization:  Sparse is {savings:.1f}x FASTER than equivalent Huge Dense.")
    
    print("\n💡 INTERPRETATION:")
    print(f"   You successfully fit {NUM_EXPERTS}x parameters into a budget of {ratio:.1f}x compute.")
    print(f"   If you tried to run all {NUM_EXPERTS} experts dense, it would be {savings:.1f}x slower.")

if __name__ == "__main__":
    benchmark_integration()
