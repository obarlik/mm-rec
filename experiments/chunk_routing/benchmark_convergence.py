
"""
Benchmark: Training Convergence (Separation Efficiency)
Does the Sparse Model learn FASTER than Dense?

Scenario:
- 4 Distinct Tasks (Task 0, 1, 2, 3), represented by orthogonal input clusters.
- Target: Regression (y = TaskParams[i] @ x).
- Model 1: Dense Shared (128 units). Must learn ALL tasks in one set of weights (Interference).
- Model 2: Sparse LSH (64 experts * 128 units). Should route each task to unique experts (Specialization).

Hypothesis: Sparse will converge significantly faster because updates don't overwrite each other (Gradient Independence).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mm_rec.blocks.sparse_mm_rec_block import LSHRouter

def benchmark_convergence():
    print("📉 BENCHMARK: Training Convergence (Sparse vs Dense)")
    print("--------------------------------------------------")
    
    dim = 64
    batch_size = 32
    steps = 500
    
    # 1. Dataset Generation (4 Orthogonal Tasks)
    torch.manual_seed(42)
    # Task Centroids (Inputs cluster around these)
    centroids = torch.randn(4, dim) * 1.5 # Lower scale from 5.0 to 1.5
    # Task Generators (Ground Truth Weights per task)
    ground_truth_weights = torch.randn(4, dim, 1) * 0.5 # Lower weight scale
    
    def get_batch():
        # Pick random tasks
        task_ids = torch.randint(0, 4, (batch_size,))
        # Generate Inputs: Centroid + Noise
        x = centroids[task_ids] + torch.randn(batch_size, dim) * 0.5
        # Generate Targets: x @ ground_truth
        # Need to gather weights: [B, D, 1]
        w = ground_truth_weights[task_ids]
        y = torch.bmm(x.unsqueeze(1), w).squeeze(-1) # [B, 1]
        return x, y, task_ids

    # 2. Models
    
    # A. Dense Model (Shared weights)
    class DenseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)
            
    # B. Sparse Model (LSH Router -> Experts)
    # Simplified for benchmark (No chunking needed for this vector-level test)
    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 16 # Keep it small but > 4
            import math
            bits = int(math.log2(self.num_experts))
            self.router = LSHRouter(dim, self.num_experts, num_bits=bits)
            # Experts: 16 linear layers
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, 128), 
                    nn.ReLU(), 
                    nn.Linear(128, 1)
                ) for _ in range(self.num_experts)
            ])
            
        def forward(self, x):
            # x: [B, D]
            # Fake chunk dim for router: [B, 1, 1, D]
            x_routed = x.view(batch_size, 1, 1, dim)
            idx, gates, _ = self.router(x_routed) # [B, 1, 2]
            
            # Simple average of heavy hitters for this demo
            top1_idx = idx[:, 0, 0] # [B]
            
            # Gather outputs (Slow loop for demo, but logically correct)
            outputs = []
            for i in range(batch_size):
                curr_exp = top1_idx[i].item()
                outputs.append(self.experts[curr_exp](x[i:i+1]))
                
            return torch.cat(outputs, dim=0)

    # 3. Training Loop
    model_dense = DenseModel()
    model_sparse = SparseModel()
    
    opt_dense = optim.SGD(model_dense.parameters(), lr=0.005)
    opt_sparse = optim.SGD(model_sparse.parameters(), lr=0.005)
    
    loss_fn = nn.MSELoss()
    
    history_dense = []
    history_sparse = []
    
    print("Training...")
    for step in range(steps):
        x, y, _ = get_batch()
        
        # Dense
        opt_dense.zero_grad()
        pred_d = model_dense(x)
        loss_d = loss_fn(pred_d, y)
        loss_d.backward()
        opt_dense.step()
        history_dense.append(loss_d.item())
        
        # Sparse
        opt_sparse.zero_grad()
        pred_s = model_sparse(x)
        loss_s = loss_fn(pred_s, y)
        loss_s.backward()
        opt_sparse.step()
        history_sparse.append(loss_s.item())
        
        if step % 50 == 0:
            print(f"  Step {step}: Dense={loss_d.item():.4f}, Sparse={loss_s.item():.4f}")

    # Results
    final_dense = np.mean(history_dense[-10:])
    final_sparse = np.mean(history_sparse[-10:])
    
    speedup = final_dense / final_sparse
    
    print("\n🏁 Results (Final Loss):")
    print(f"  Dense Shared: {final_dense:.4f}")
    print(f"  Sparse LSH:   {final_sparse:.4f}")
    
    if final_sparse < final_dense:
        print(f"  ✅ Sparse converged to {speedup:.1f}x lower loss (Separation Efficiency).")
        print("  Reason: Experts specialized to tasks without fighting each other.")
    else:
        print("  ❌ Sparse is slower/worse (Maybe router struggles?).")

if __name__ == "__main__":
    benchmark_convergence()
