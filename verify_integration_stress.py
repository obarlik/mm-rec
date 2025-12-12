
"""
Integration Falsification: Real Model Stress Test
We verified the 'Router' in isolation. Now we verify the 'MMRecModel' itself.
Does the logic hold up when buried deep inside the 24-layer beast?

Tests:
1. Connectivity: Can we reach the SparseFFN layers?
2. Routing Verification: Do they actually perform Top-2 routing on input data?
3. Gradient Flow: Do gradients survive the sparse Top-2 path?
"""

import torch
import torch.nn as nn
from mm_rec.model import MMRecModel

def verify_integration_stress():
    print("🕵️  Starting Integration Falsification (The Real Deal)...")
    
    # 1. Instantiate the Full Model (Sparse Config)
    config = {
        "vocab_size": 1000,
        "model_dim": 128,
        "num_layers": 2, # Keep it small for debugging
        "use_sparse": True,
        "sparse_chunk_size": 16,
        "num_experts": 4 # Power of 2
    }
    
    model = MMRecModel(**config)
    print("    Model Instantiated (Sparse Mode).")
    
    # 2. Hook into the Sparse Layer
    # We want to see what the Router is actually doing.
    # We'll attach a hook to the first layer's FFN Router.
    
    router_logs = []
    
    def hook_fn(module, input, output):
        # input is a tuple, input[0] is the tensor passed to forward
        # output of Router is (expert_indices, gates, confidence)
        indices, gates, confidence = output
        router_logs.append({
            "indices_shape": indices.shape, # [B, NumChunks, TopK]
            "indices_sample": indices[0, 0, :].detach().tolist(), # First chunk
            "gates_mean": gates.mean().item()
        })
        
    # Find the first SparseFFN router
    # Structure: model.blocks[0].ffn.router
    target_router = model.blocks[0].ffn.router
    handle = target_router.register_forward_hook(hook_fn)
    print("    Hook attached to Layer 0 Router.")
    
    # 3. Feed Data (Forward Pass)
    x = torch.randint(0, 1000, (2, 64)) # Batch 2, Seq 64
    
    print("    Running Forward Pass...")
    out = model(x)
    
    # 4. Check Hook Data
    if len(router_logs) > 0:
        log = router_logs[0]
        print(f"    CAPTURED ROUTING INFO:")
        print(f"    - Indices Shape: {log['indices_shape']} (Expected [2, 4, 2])")
        print(f"    - Top-2 Sample: {log['indices_sample']}")
        
        # Validation
        if log['indices_shape'][-1] == 2:
            print("    ✅ PASS: System is actively using Top-2 Routing inside the model.")
        else:
            print("    ❌ FAIL: Router is not outputting Top-2 indices.")
    else:
        print("    ❌ FAIL: Hook did not capture anything. Is the layer running?")
        
    # 5. Gradient Test (The ultimate proof of connectivity)
    print("\n    Running Gradient Test...")
    loss = out.sum()
    loss.backward()
    
    # Check gradients on the EXPERT WEIGHTS of that layer
    # model.blocks[0].ffn.expert_weights
    expert_grads = model.blocks[0].ffn.expert_weights.grad
    
    if expert_grads is not None:
        grad_norm = expert_grads.norm().item()
        print(f"    Expert Weights Grad Norm: {grad_norm:.4f}")
        if grad_norm > 0:
            print("    ✅ PASS: Gradients are flowing to the sparse experts!")
        else:
            print("    ⚠️  WARNING: Gradients exist but are zero. (Could be random chance or dead neurons)")
    else:
        print("    ❌ FAIL: No gradients on Expert Weights. Graph is broken.")

    # ... (Previous Grad Test) ...
    
    # 6. Threshold / Creative Mode Test
    print("\n    Running Uncertainty Threshold Test...")
    # Run with HIGH threshold -> Should block experts (Creative Mode OFF)
    # The output should change (likely degrade, or just be different)
    
    # Reset Grads or just run forward
    out_creative = model(x, router_threshold=0.0) # Creative (Default)
    out_strict = model(x, router_threshold=100.0) # Strict (Impossible confidence)
    
    diff = (out_creative - out_strict).abs().sum().item()
    print(f"    Diff between Threshold 0.0 and 100.0: {diff:.4f}")
    
    if diff > 0.0:
        print("    ✅ PASS: Thresholding changes the output (Control works).")
    else:
        print("    ❌ FAIL: Thresholding had no effect. (Did we pass the arg?)")

    print("\n🏁 Integration Falsification Complete.")

if __name__ == "__main__":
    verify_integration_stress()
