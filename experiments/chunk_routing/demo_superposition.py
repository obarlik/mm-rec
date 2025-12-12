
"""
Demo: Superposition Limit (Scaling Multidisciplinary)
What happens if we mix 3, 4, or 5 topics?
Does the LSH Router create meaningful niches or collapse to noise?

Hypothesis:
- 2-3 Topics: Robust Niche (Specific combination expert)
- 5+ Topics: White Noise (Central Limit Theorem) -> Random Expert or "Generalist" Expert?
"""

import torch
import numpy as np
import sys
import os

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mm_rec.blocks.sparse_mm_rec_block import LSHRouter

def demo_superposition():
    print("🌌 DEMO: Superposition Limit (3+ Fields)")
    print("----------------------------------------")
    
    dim = 256 # Higher dim to support more orthogonality
    num_experts = 64
    router = LSHRouter(dim, num_experts)
    
    # 1. Generate 5 Orthogonal Topics
    # We use QR decomposition to guarantee they are mathematically distinct (angle=90 deg)
    torch.manual_seed(101)
    base = torch.randn(dim, 5)
    q, _ = torch.linalg.qr(base) # Q is [dim, 5] orthogonal columns
    topics = [q[:, i].view(1, 1, 1, dim) * 10.0 for i in range(5)] # Scale up strength
    
    topic_names = ["Physics", "History", "Art", "Coding", "Cooking"]
    
    print("Dimensions initialized (Orthogonal).")
    
    # Check individual mappings
    topic_experts = {}
    for name, vec in zip(topic_names, topics):
        idx, _ = router(vec)
        exp = idx[0,0,0].item()
        topic_experts[name] = exp
        print(f"  - {name:<8} -> Expert {exp}")

    # 2. Progressive Mixing
    print("\n🧪 Progressive Mixing Tests:")
    
    current_mix = torch.zeros_like(topics[0])
    current_names = []
    
    for i in range(5):
        # Add next topic
        current_mix += topics[i]
        current_names.append(topic_names[i])
        
        mix_name = "+".join(current_names)
        
        # Route
        idx_mix, _ = router(current_mix)
        chosen = idx_mix[0,0].tolist() # Top-2
        
        print(f"\n[{i+1} Fields] {mix_name}")
        print(f"   -> Routes to: {chosen}")
        
        # Check if chosen experts are among the constituents' original experts
        constituent_experts = [topic_experts[n] for n in current_names]
        matches = [c for c in chosen if c in constituent_experts]
        
        if matches:
            print(f"   ✅ Recall: Retained expert(s) from constituents: {matches}")
        else:
            print(f"   🆕 Niche: Assigned to a completely new 'Composite' expert.")
            
    # 3. Noise Comparison
    print("\n📉 White Noise Comparison:")
    # What does pure random noise route to?
    noise = torch.randn(1, 1, 1, dim)
    idx_noise, _ = router(noise)
    print(f"   Pure Noise -> {idx_noise[0,0].tolist()}")
    
    if idx_noise[0,0].tolist() == chosen:
         print("   ⚠️  WARNING: The 5-field mix looks just like random noise to the router.")
    else:
         print("   ✅ The 5-field mix is distinct from random noise.")

if __name__ == "__main__":
    demo_superposition()
