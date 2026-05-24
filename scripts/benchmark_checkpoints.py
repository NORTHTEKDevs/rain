# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Apples-to-apples A/B benchmark of two HYMN-Plus checkpoints on the same
held-out corpus.

Runs eval_hymn_plus.evaluate_nll on each checkpoint with identical settings,
then prints a side-by-side table + winner verdict.

Usage:
    python -m scripts.benchmark_checkpoints \
        --checkpoint-a data/checkpoints/hymn_plus_v1_5k.npz \
        --checkpoint-b data/checkpoints/hymn_plus_v2_15k.npz \
        --corpus data/corpora/wikitext2_train.txt \
        --n-eval-chars 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval_hymn_plus import _load_checkpoint, evaluate_nll


def _eval_one(ckpt_path: Path, text: str, batch_size: int, seq_len: int, device: str) -> dict:
    model, char_to_id, meta = _load_checkpoint(ckpt_path)
    res = evaluate_nll(
        model, char_to_id, text, batch_size=batch_size, seq_len=seq_len, device=device
    )
    res["params"] = meta.get("trainable_params")
    res["dim"] = meta.get("dim")
    res["n_layers"] = meta.get("n_layers")
    res["train_steps"] = meta.get("steps")
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--n-eval-chars", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")[: args.n_eval_chars]
    print(f"evaluating both checkpoints on {len(text):,} chars of {args.corpus}")
    print()

    res_a = _eval_one(Path(args.checkpoint_a), text, args.batch_size, args.seq_len, args.device)
    res_b = _eval_one(Path(args.checkpoint_b), text, args.batch_size, args.seq_len, args.device)

    if "error" in res_a or "error" in res_b:
        print(json.dumps({"a": res_a, "b": res_b}, indent=2))
        return 1

    def _row(name: str, r: dict) -> str:
        return (
            f"{name:24s} | params={r.get('params', '?'):>10} | "
            f"dim={r.get('dim', '?'):>4} | layers={r.get('n_layers', '?'):>2} | "
            f"steps={r.get('train_steps', '?'):>6} | "
            f"NLL={r['nll_nats_per_char']:.4f} | "
            f"PPL={r['perplexity']:.2f} | "
            f"%uniform={r['ratio_of_uniform'] * 100:.1f}"
        )

    print(_row("A: " + Path(args.checkpoint_a).name, res_a))
    print(_row("B: " + Path(args.checkpoint_b).name, res_b))
    print()
    diff = res_a["nll_nats_per_char"] - res_b["nll_nats_per_char"]
    if abs(diff) < 0.01:
        verdict = "TIE (within 0.01 NLL)"
    elif diff > 0:
        verdict = f"B wins by {diff:.4f} NLL ({diff / res_a['nll_nats_per_char'] * 100:.1f}% better than A)"
    else:
        verdict = f"A wins by {-diff:.4f} NLL ({-diff / res_b['nll_nats_per_char'] * 100:.1f}% better than B)"
    print(f"verdict: {verdict}")

    summary = {
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "corpus": str(args.corpus),
        "chars_evaluated": res_a["chars_evaluated"],
        "results_a": res_a,
        "results_b": res_b,
        "diff_nll": diff,
        "verdict": verdict,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nfull JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
