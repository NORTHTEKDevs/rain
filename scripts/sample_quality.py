"""Sample-quality eval for a HYMN-Plus checkpoint.

Generates N samples from a fixed prompt, then computes:

  * verbatim_overlap_pct  -- longest-substring-from-training metric.
    Slides a 32-char window through each sample; reports the fraction
    of windows that appear verbatim in the training corpus. High value
    = memorization risk.

  * unique_window_pct -- fraction of 16-char windows that are unique
    across the N samples. Low value = mode collapse / repetition.

  * mean_distinct_2gram, mean_distinct_3gram -- diversity within each
    sample (higher = less repetitive).

Use this after every training run to catch:
  - Memorization (verbatim_overlap high)
  - Mode collapse (unique_window_pct low)
  - Degenerate looping (distinct n-grams low)

Usage:
    python -m scripts.sample_quality \
        --checkpoint data/checkpoints/hymn_plus_v1_5k.npz \
        --train-corpus data/corpora/tiny_shakespeare.txt \
        --prompts "ROMEO: " "Once" "The king" \
        --n-samples 8 --max-new 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rain.cognition.hymn_plus_sampler import HymnPlusSampler


def _ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _verbatim_overlap_pct(sample: str, corpus: str, window: int = 32) -> float:
    """Fraction of `window`-char sliding sub-strings that appear verbatim in corpus."""
    if len(sample) < window:
        return 0.0
    n_windows = len(sample) - window + 1
    hits = sum(1 for i in range(n_windows) if sample[i : i + window] in corpus)
    return hits / max(1, n_windows)


def _unique_window_pct(samples: list[str], window: int = 16) -> float:
    """Across all samples, fraction of `window`-char sub-strings that are unique."""
    all_windows: list[str] = []
    for s in samples:
        if len(s) >= window:
            all_windows.extend(s[i : i + window] for i in range(len(s) - window + 1))
    if not all_windows:
        return 0.0
    return len(set(all_windows)) / len(all_windows)


def _distinct_ngram_ratio(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams)


def assess(
    checkpoint: str,
    train_corpus_path: str,
    prompts: list[str],
    *,
    n_samples: int = 8,
    max_new: int = 200,
    temperature: float = 0.7,
    top_k: int = 30,
) -> dict:
    corpus_text = Path(train_corpus_path).read_text(encoding="utf-8")
    sampler = HymnPlusSampler.from_checkpoint(checkpoint, temperature=temperature, top_k=top_k)

    by_prompt = {}
    all_samples: list[str] = []
    for prompt in prompts:
        samples = []
        for i in range(n_samples):
            torch.manual_seed(1000 + i)  # deterministic but diverse
            samples.append(prompt + sampler(prompt, n_tokens=max_new))
        overlap = [_verbatim_overlap_pct(s, corpus_text) for s in samples]
        d2 = [_distinct_ngram_ratio(s, 2) for s in samples]
        d3 = [_distinct_ngram_ratio(s, 3) for s in samples]
        by_prompt[prompt] = {
            "samples": samples,
            "mean_verbatim_overlap_pct": round(sum(overlap) / len(overlap) * 100, 2),
            "max_verbatim_overlap_pct": round(max(overlap) * 100, 2),
            "mean_distinct_2gram": round(sum(d2) / len(d2), 4),
            "mean_distinct_3gram": round(sum(d3) / len(d3), 4),
            "unique_window_pct_within_prompt": round(_unique_window_pct(samples) * 100, 2),
        }
        all_samples.extend(samples)

    return {
        "checkpoint": checkpoint,
        "train_corpus": train_corpus_path,
        "n_prompts": len(prompts),
        "n_samples_per_prompt": n_samples,
        "max_new": max_new,
        "temperature": temperature,
        "top_k": top_k,
        "global_unique_window_pct": round(_unique_window_pct(all_samples) * 100, 2),
        "global_mean_verbatim_overlap_pct": round(
            sum(p["mean_verbatim_overlap_pct"] for p in by_prompt.values()) / len(by_prompt), 2
        ),
        "by_prompt": by_prompt,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Sample-quality eval (memorization + diversity)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-corpus", required=True, help="for verbatim overlap detection")
    p.add_argument(
        "--prompts",
        nargs="+",
        default=["ROMEO: ", "Once upon", "The king"],
        help="prompts to sample from",
    )
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = assess(
        args.checkpoint,
        args.train_corpus,
        args.prompts,
        n_samples=args.n_samples,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Summary line
    print("=== sample-quality summary ===")
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"verbatim-overlap (avg of {args.n_samples}x{len(args.prompts)}): "
        f"{res['global_mean_verbatim_overlap_pct']:.2f}%  "
        "(HIGH = memorization)"
    )
    print(
        f"unique 16-char windows globally: {res['global_unique_window_pct']:.2f}%  "
        "(LOW = mode collapse)"
    )
    for prompt, data in res["by_prompt"].items():
        print(f"\nprompt: {prompt!r}")
        print(
            f"  mean overlap: {data['mean_verbatim_overlap_pct']:.2f}%  "
            f"max: {data['max_verbatim_overlap_pct']:.2f}%"
        )
        print(
            f"  distinct 2-grams: {data['mean_distinct_2gram']:.3f}  "
            f"3-grams: {data['mean_distinct_3gram']:.3f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nfull JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
