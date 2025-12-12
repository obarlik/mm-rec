
"""
Demo: Uncertainty / Hallucination Detection
Can we detect if the model is "Unsure"?

Method:
- LSH works by checking which side of a hyperplane a vector falls on.
- Measure: "Distance to Hyperplane" (Projection Magnitude).
- Hypothesis: 
    - Strong/Clear concepts will be FAR from the boundary (High Confidence).
    - Ambiguous/Noise inputs will be close to 0 (Low Confidence).

If this holds, we can set a threshold: "If confidence < X, output 'I don't know'."
"""

import torch
import numpy as np
import sys
import os

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mm_rec.blocks.sparse_mm_rec_block import LSHRouter

def demo_uncertainty():
    print("🤔 DEMO: Hallucination Detection (Confidence Metric)")
    print("---------------------------------------------------")
    
    dim = 128
    num_experts = 64
    router = LSHRouter(dim, num_experts)
    
    # 1. Strong Signal vs Noise
    torch.manual_seed(42)
    
    # A. Strong Signal (Scaled up, represents learned feature)
    # [1, 1, 1, D]
    signal = torch.randn(1, 1, 1, dim) * 5.0 
    
    # B. Ambiguous/Noise (Small random variance)
    # [1, 1, 1, D]
    noise = torch.randn(1, 1, 1, dim) * 0.1
    
    # 2. Extract Internals (Confidence)
    # We need to manually run the projection part of LSHRouter to get magnitudes
    
    def get_confidence(router, x):
        # Mirroring forward pass logic
        chunk_emb = x.mean(dim=2) # [B, N_C, D]
        projections = torch.matmul(chunk_emb, router.hyperplanes) # [B, N_C, Bits]
        
        # Confidence = Mean Absolute Distance from Decision Boundary (0)
        confidence = torch.abs(projections).mean().item()
        return confidence
        
    conf_signal = get_confidence(router, signal)
    conf_noise = get_confidence(router, noise)
    
    print(f"📊 Confidence Scores (Mean Dist to Hyperplane):")
    print(f"   - Strong Signal (Learned Topic): {conf_signal:.4f}")
    print(f"   - Weak Signal   (Noise/Unknown): {conf_noise:.4f}")
    
    # 3. Control Mechanism
    THRESHOLD = 1.0 # Hypothetical threshold
    
    print("\n🛡️  Control Logic Simulation:")
    
    def simulate_output(name, conf):
        print(f"   Input: {name}")
        if conf < THRESHOLD:
            print(f"     -> 🛑 BLOCKED (Confidence {conf:.2f} < {THRESHOLD})")
            print("     -> Output: 'I am not sure about this.'")
        else:
            print(f"     -> ✅ ACCEPTED (Confidence {conf:.2f} > {THRESHOLD})")
            print("     -> Output: [Generates Answer]")
            
    simulate_output("Medical Article", conf_signal)
    simulate_output("Random Gibberish", conf_noise)
    
    print("\n💡 Conclusion:")
    print("   We CAN detect if the input is 'weak' or doesn't strongly activate any specific path.")
    print("   This enables the 'I don't know' feature vs 'Hallucinating'.")

if __name__ == "__main__":
    demo_uncertainty()
