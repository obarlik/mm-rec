"""
PCGrad-like prototype without extra dependencies.

Two-head toy MLP with two losses:
- Baseline: standard sum of losses
- PCGrad-like: project gradients to reduce conflicts (dot<0)

Metrics:
- Final losses and time per step
- Saved as JSON under experiments/results/pcgrad.json
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import torch
from torch import nn

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "pcgrad.json"
)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


class TwoHeadMLP(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128, out1: int = 16, out2: int = 8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(hidden, out1)
        self.head2 = nn.Linear(hidden, out2)

    def forward(self, x):
        h = self.shared(x)
        return self.head1(h), self.head2(h)


def pcgrad_step(model: nn.Module, optimizer, x, y1, y2):
    optimizer.zero_grad()
    out1, out2 = model(x)
    loss1 = torch.nn.functional.mse_loss(out1, y1)
    loss2 = torch.nn.functional.mse_loss(out2, y2)

    params = [p for p in model.parameters() if p.requires_grad]
    g1 = torch.autograd.grad(loss1, params, retain_graph=True, allow_unused=False)
    g2 = torch.autograd.grad(loss2, params, allow_unused=False)

    proj_grads: List[torch.Tensor] = []
    for g1_p, g2_p in zip(g1, g2):
        # Flatten to compute dot
        g1_flat = g1_p.view(-1)
        g2_flat = g2_p.view(-1)
        dot = torch.dot(g1_flat, g2_flat)
        if dot < 0:
            # project g1 onto orthogonal of g2: g1 - (dot/||g2||^2) * g2
            denom = torch.dot(g2_flat, g2_flat) + 1e-12
            g1_p = g1_p - (dot / denom) * g2_p
        proj_grads.append(g1_p + g2_p)

    with torch.no_grad():
        for p, g in zip(params, proj_grads):
            p.grad = g
    optimizer.step()
    return loss1.item(), loss2.item()


def baseline_step(model: nn.Module, optimizer, x, y1, y2):
    optimizer.zero_grad()
    out1, out2 = model(x)
    loss1 = torch.nn.functional.mse_loss(out1, y1)
    loss2 = torch.nn.functional.mse_loss(out2, y2)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    return loss1.item(), loss2.item()


def run_experiment(kind: str, steps: int = 200, dim: int = 64, device: str = "cpu"):
    torch.manual_seed(0)
    model = TwoHeadMLP(dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses1: List[float] = []
    losses2: List[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        x = torch.randn(128, dim, device=device)
        y1 = torch.randn(128, 16, device=device)
        y2 = torch.randn(128, 8, device=device)
        if kind == "pcgrad":
            l1, l2 = pcgrad_step(model, optimizer, x, y1, y2)
        else:
            l1, l2 = baseline_step(model, optimizer, x, y1, y2)
        losses1.append(l1)
        losses2.append(l2)
    elapsed = time.perf_counter() - start
    return {
        "kind": kind,
        "steps": steps,
        "final_loss1": losses1[-1],
        "final_loss2": losses2[-1],
        "time_per_step_ms": (elapsed / steps) * 1000,
        "mean_loss1": sum(losses1) / len(losses1),
        "mean_loss2": sum(losses2) / len(losses2),
    }


def main():
    device = "cpu"
    results = [
        run_experiment("baseline", device=device),
        run_experiment("pcgrad", device=device),
    ]
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Results written to", RESULTS_PATH)
    for r in results:
        print(
            f"{r['kind']:>8} | loss1={r['final_loss1']:.4f} "
            f"loss2={r['final_loss2']:.4f} | t/step={r['time_per_step_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
