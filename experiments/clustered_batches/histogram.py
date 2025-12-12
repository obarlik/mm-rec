"""
Length histogram and padding estimator.

Usage:
  python experiments/clustered_batches/histogram.py --batch-size 32
  python experiments/clustered_batches/histogram.py --lengths-file /path/to/lengths.txt

If --lengths-file verilmezse sentetik uzunluklar üretir.
Lengths dosyası beklentisi: her satırda tek bir tamsayı (token sayısı) veya
metin satırları (boşluk sayımıyla uzunluk çıkarılır).
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import List

import torch


def load_lengths(path: str) -> List[int]:
    lengths: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # integer line or text line
            try:
                val = int(line)
                lengths.append(val)
            except ValueError:
                lengths.append(len(line.split()))
    return lengths


def synthetic_lengths(n: int = 2000, min_len: int = 64, max_len: int = 512) -> List[int]:
    rng = torch.Generator().manual_seed(0)
    return torch.randint(min_len, max_len + 1, (n,), generator=rng).tolist()


def padding_ratio(lengths: List[int], batch_size: int, sort: bool) -> float:
    if sort:
        lengths = sorted(lengths)
    total_pad = 0
    total_tokens = 0
    for i in range(0, len(lengths), batch_size):
        chunk = lengths[i : i + batch_size]
        if not chunk:
            continue
        max_len = max(chunk)
        total_tokens += max_len * len(chunk)
        total_pad += (max_len * len(chunk)) - sum(chunk)
    return (total_pad / total_tokens) if total_tokens > 0 else 0.0


def summarize(lengths: List[int], batch_size: int):
    avg = statistics.mean(lengths)
    median = statistics.median(lengths)
    p90 = float(torch.tensor(lengths).float().kthvalue(int(0.9 * len(lengths)) + 1).values.item())
    pad_random = padding_ratio(lengths, batch_size, sort=False)
    pad_sorted = padding_ratio(lengths, batch_size, sort=True)
    return {
        "count": len(lengths),
        "mean": avg,
        "median": median,
        "p90": p90,
        "padding_random": pad_random,
        "padding_sorted": pad_sorted,
        "padding_saved_by_sort": pad_random - pad_sorted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths-file", type=str, default=None, help="Path to lengths.txt or raw text")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.lengths_file:
        lengths = load_lengths(args.lengths_file)
    else:
        lengths = synthetic_lengths()

    stats = summarize(lengths, args.batch_size)
    print(f"Samples: {stats['count']}")
    print(f"mean={stats['mean']:.1f} median={stats['median']:.1f} p90={stats['p90']:.1f}")
    print(f"padding random={stats['padding_random']:.3f} sorted={stats['padding_sorted']:.3f} "
          f"saving={stats['padding_saved_by_sort']:.3f}")


if __name__ == "__main__":
    main()
