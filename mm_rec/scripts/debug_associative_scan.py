#!/usr/bin/env python3
"""Debug Associative Scan - Doğruluk sorununu bul"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "mm_rec" / "cpp"))

import mm_rec_scan_cpu

# Small test case
batch, heads, seq_len, dim = 1, 1, 8, 4
gamma = torch.rand(batch, heads, seq_len, dim, dtype=torch.float32)
gamma = torch.clamp(gamma, 0.1, 0.9)

print("Input gamma:")
print(gamma[0, 0])

# Reference (PyTorch Sequential)
log_gamma = torch.log(gamma + 1e-8)
log_gamma = torch.clamp(log_gamma, -50.0, 0.0)

log_cumsum_ref = torch.zeros_like(log_gamma)
log_cumsum_ref[:, :, 0, :] = log_gamma[:, :, 0, :]

for t in range(1, seq_len):
    prev = log_cumsum_ref[:, :, t-1, :]
    curr = log_gamma[:, :, t, :]
    max_val = torch.max(prev, curr)
    diff = torch.abs(prev - curr)
    diff_clamped = torch.clamp(diff, max=20.0)
    log_cumsum_ref[:, :, t, :] = max_val + torch.log1p(torch.exp(-diff_clamped))

max_log = torch.max(log_cumsum_ref, dim=2, keepdim=True)[0]
stable_log = log_cumsum_ref - max_log
result_ref = torch.exp(stable_log) * torch.exp(max_log)

print("\nReference (PyTorch Sequential) log_cumsum:")
print(log_cumsum_ref[0, 0])
print("\nReference result:")
print(result_ref[0, 0])

# C++ Result
result_cpp = mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)

print("\nC++ Result:")
print(result_cpp[0, 0])

print("\nDifference:")
print(torch.abs(result_ref - result_cpp)[0, 0])

print("\nMax difference:", torch.max(torch.abs(result_ref - result_cpp)).item())
