# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Sweep temperature x top-k for a HYMN-Plus checkpoint; pick the setting
with the best (low verbatim-overlap, high distinct-3gram) trade-off.

Use this once after each new checkpoint to find production sampling
defaults rather than guessing.

Usage:
    python -m scripts.calibrate_sampler \
        --checkpoint data/checkpoints/hymn_plus_v1_5k.npz \
        --train-corpus data/corpora/tiny_shakespeare.txt \
        --prompt "ROMEO: " --n-samples 4 --max-new 120
"""

from __future__ import annotations

import argparse
import json
from itertools import product

from rain.cognition.hymn_plus_sampler import HymnPlusSampler
from scripts.sample_quality import _distinct_ngram_ratio, _verbatim_overlap_pct


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-corpus", required=True)
    p.add_argument("--prompt", default="ROMEO: ")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-new", type=int, default=120)
    p.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9, 1.1],
    )
    p.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=[10, 30, 100, 0],
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from pathlib import Path

    corpus = Path(args.train_corpus).read_text(encoding="utf-8")

    grid = []
    for temp, top_k in product(args.temps, args.top_ks):
        sampler = HymnPlusSampler.from_checkpoint(args.checkpoint, temperature=temp, top_k=top_k)
        samples = []
        for i in range(args.n_samples):
            import torch

            torch.manual_seed(1000 + i)
            samples.append(args.prompt + sampler(args.prompt, n_tokens=args.max_new))
        overlap = sum(_verbatim_overlap_pct(s, corpus) for s in samples) / args.n_samples
        d3 = sum(_distinct_ngram_ratio(s, 3) for s in samples) / args.n_samples
        # Combined score: maximize diversity - 2x penalize memorization
        # (memorization is the worse failure mode in production).
        score = d3 - 2.0 * overlap
        grid.append(
            {
                "temperature": temp,
                "top_k": top_k,
                "mean_verbatim_overlap_pct": round(overlap * 100, 2),
                "mean_distinct_3gram": round(d3, 4),
                "score": round(score, 4),
            }
        )

    grid.sort(key=lambda r: r["score"], reverse=True)

    print(f"{'temp':>5} {'top-k':>6} {'overlap%':>9} {'distinct3':>10} {'score':>7}")
    for row in grid:
        print(
            f"{row['temperature']:>5.2f} {row['top_k']:>6} "
            f"{row['mean_verbatim_overlap_pct']:>8.2f}% "
            f"{row['mean_distinct_3gram']:>10.4f} "
            f"{row['score']:>7.4f}"
        )
    best = grid[0]
    print(
        f"\nrecommended sampling: temperature={best['temperature']} top_k={best['top_k']} "
        f"(overlap {best['mean_verbatim_overlap_pct']:.2f}%, "
        f"distinct-3 {best['mean_distinct_3gram']:.4f})"
    )

    if args.out:
        Path(args.out).write_text(json.dumps({"grid": grid, "recommended": best}, indent=2))
        print(f"\nfull JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
