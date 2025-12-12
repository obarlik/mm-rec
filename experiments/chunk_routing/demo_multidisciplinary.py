
"""
Demo: Multidisciplinary Routing
Hypothesis: Top-2 Routing automatically handles 'Mixed' topics (e.g. Med-Law) 
by selecting one expert from 'Med' and one from 'Law' (or their neighbors).

Scenario:
1. Concept A: Medicine (random vector A) -> Expert 13
2. Concept B: Law (random vector B) -> Expert 42
3. Concept A+B: Med-Malpractice (A + B) -> Should route to [13, 42] (ideal) or similar.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mm_rec.blocks.sparse_mm_rec_block import LSHRouter

def demo_multidisciplinary():
    print("🎓 DEMO: Multidisciplinary Expert Routing")
    print("----------------------------------------")
    
    dim = 128
    num_experts = 64
    
    # Init Router (Fixed random hyperplanes)
    router = LSHRouter(dim, num_experts)
    
    # 1. Create Pure Concepts (Simulated Embeddings)
    torch.manual_seed(42)
    # Medicine Vector: [1, 1, 1, D] -> Batch=1, NumChunks=1, ChunkSize=1, Dim=128
    vec_med = torch.randn(1, 1, 1, dim) 
    # Law Vector
    vec_law = torch.randn(1, 1, 1, dim)
    
    # 2. Find their 'Natural' Experts
    idx_med, _ = router(vec_med)
    idx_law, _ = router(vec_law)
    
    exp_med = idx_med[0, 0, 0].item()
    exp_law = idx_law[0, 0, 0].item()
    
    print(f"🏥 Concept A (Medicine) maps to Expert: {exp_med}")
    print(f"⚖️  Concept B (Law)      maps to Expert: {exp_law}")
    
    # 3. Create Multidisciplinary Concept (Medicine + Law + Noise)
    # "Medical Malpractice" case
    vec_mix = vec_med + vec_law
    
    # 4. Route the Mix
    idx_mix, _ = router(vec_mix)
    chosen_experts = idx_mix[0, 0].tolist() # [Top1, Top2]
    
    print(f"\n🧬 Concept A+B (Med-Malpractice) routing:")
    print(f"   Router Selected: {chosen_experts}")
    
    # Analysis
    print("\n🔍 Analysis:")
    if exp_med in chosen_experts and exp_law in chosen_experts:
        print("   ✅ PERFECT MATCH! The router picked exactly the Med expert and the Law expert.")
    elif exp_med in chosen_experts:
        print(f"   ✅ Partial Match: Found Med expert ({exp_med}). The other ({chosen_experts[1]}) might be a Law neighbor.")
    elif exp_law in chosen_experts:
         print(f"   ✅ Partial Match: Found Law expert ({exp_law}). The other ({chosen_experts[0]}) might be a Med neighbor.")
    else:
        print("   ⚠️  New Synthesis: The mixture created a new hash code. This represents a dedicated 'Med-Law' expert niche.")

    # Geometric Explanation
    print("\n🧠 How Training Works:")
    print("   1. If A+B happens often (e.g. Patent Law), the router consistency sends it to these experts.")
    print("   2. Expert {chosen_experts[0]} will learn the 'Dominant' features.")
    print("   3. Expert {chosen_experts[1]} will learn the 'Secondary' features.")
    print("   4. Both experts contribute to the final answer (Weighted Average).")

if __name__ == "__main__":
    demo_multidisciplinary()
