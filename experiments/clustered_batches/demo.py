"""
Clustered batching prototype (no extra deps).

Scenarios compared:
- random batching
- length bucketing (sort by sequence length)
- embedding + length greedy clustering (cosine similarity)

Outputs:
- Prints padding ratio stats and step times.
- Writes metrics to JSON under experiments/results/clustered_batches.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "clustered_batches.json"
)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


@dataclass
class Sample:
    embedding: torch.Tensor  # [d]
    length: int


def generate_samples(
    num_samples: int = 2000, emb_dim: int = 64, min_len: int = 64, max_len: int = 512
) -> List[Sample]:
    rng = torch.Generator().manual_seed(0)
    lengths = torch.randint(min_len, max_len + 1, (num_samples,), generator=rng).tolist()
    embeddings = torch.randn(num_samples, emb_dim, generator=rng)
    samples = [Sample(embeddings[i], int(lengths[i])) for i in range(num_samples)]
    return samples


def padding_stats(batches: List[List[Sample]]) -> Tuple[float, float]:
    """Returns (avg_padding_ratio, max_padding_ratio)."""
    ratios = []
    for batch in batches:
        if not batch:
            continue
        max_len = max(s.length for s in batch)
        total = max_len * len(batch)
        actual = sum(s.length for s in batch)
        ratios.append((total - actual) / total if total > 0 else 0.0)
    if not ratios:
        return 0.0, 0.0
    return float(sum(ratios) / len(ratios)), float(max(ratios))


def random_batches(samples: Sequence[Sample], batch_size: int, seed: int = 0):
    rnd = random.Random(seed)
    idx = list(range(len(samples)))
    rnd.shuffle(idx)
    return [ [samples[i] for i in idx[j:j+batch_size]] for j in range(0, len(idx), batch_size) ]


def length_bucket_batches(samples: Sequence[Sample], batch_size: int):
    idx = sorted(range(len(samples)), key=lambda i: samples[i].length)
    return [ [samples[i] for i in idx[j:j+batch_size]] for j in range(0, len(idx), batch_size) ]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def embedding_length_batches(samples: Sequence[Sample], batch_size: int, seed: int = 0):
    rnd = random.Random(seed)
    remaining = list(range(len(samples)))
    batches = []
    while remaining:
        pivot_idx = remaining.pop()
        pivot = samples[pivot_idx]
        if not remaining:
            batches.append([pivot])
            break
        # Compute cosine sim to all remaining (small scale, pure torch on CPU)
        sims = [
            (cosine_sim(pivot.embedding, samples[j].embedding), j) for j in remaining
        ]
        sims.sort(reverse=True)
        chosen = [pivot_idx]
        for _, j in sims:
            if len(chosen) >= batch_size:
                break
            chosen.append(j)
        # Remove chosen from remaining
        remaining = [j for j in remaining if j not in set(chosen)]
        # Keep sequences in chosen order for consistency
        batches.append([samples[j] for j in chosen])
    # Within each batch, keep a mild length sort to reduce padding
    for b in batches:
        b.sort(key=lambda s: s.length)
    return batches


def simulate_forward(batches: List[List[Sample]], emb_dim: int, device: str = "cpu"):
    """Toy workload: sum of embeddings padded to max_len; returns total tokens and elapsed."""
    total_tokens = 0
    start = time.perf_counter()
    for batch in batches:
        if not batch:
            continue
        max_len = max(s.length for s in batch)
        total_tokens += max_len * len(batch)
        # create padded tensor [B, T, D]
        B = len(batch)
        padded = torch.zeros(B, max_len, emb_dim, device=device)
        for i, s in enumerate(batch):
            L = s.length
            padded[i, :L] = s.embedding[:emb_dim].repeat(L, 1)  # synthetic token reps
        _ = padded.sum()  # lightweight op to simulate compute
    elapsed = time.perf_counter() - start
    return total_tokens, elapsed


def run_scenario(name: str, batches: List[List[Sample]], emb_dim: int, device: str):
    avg_pad, max_pad = padding_stats(batches)
    tokens, elapsed = simulate_forward(batches, emb_dim, device)
    tps = tokens / elapsed if elapsed > 0 else 0.0
    return {
        "scenario": name,
        "num_batches": len(batches),
        "avg_padding": avg_pad,
        "max_padding": max_pad,
        "total_tokens": tokens,
        "time_sec": elapsed,
        "tokens_per_sec": tps,
    }


def main():
    device = "cpu"
    batch_size = 32
    samples = generate_samples()
    emb_dim = samples[0].embedding.numel()

    scenarios = []
    scenarios.append(
        run_scenario("random", random_batches(samples, batch_size), emb_dim, device)
    )
    scenarios.append(
        run_scenario(
            "length_bucket", length_bucket_batches(samples, batch_size), emb_dim, device
        )
    )
    scenarios.append(
        run_scenario(
            "embedding_length",
            embedding_length_batches(samples, batch_size),
            emb_dim,
            device,
        )
    )

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)

    print("Results written to", RESULTS_PATH)
    for s in scenarios:
        print(
            f"{s['scenario']:>16} | batches={s['num_batches']:4d} | "
            f"avg_pad={s['avg_padding']:.3f} max_pad={s['max_padding']:.3f} | "
            f"tps={s['tokens_per_sec']:.0f}"
        )


if __name__ == "__main__":
    main()
