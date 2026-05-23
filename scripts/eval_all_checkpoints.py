# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Run L1 (Tiny Shakespeare) across every HYMN checkpoint in a directory.

Picks the right metric per checkpoint via the sidecar's loss_type +
carry_steps fields (schema v2). Emits a compact comparison table on
stdout and a JSON report of all results.

Usage:
    python -m scripts.eval_all_checkpoints \\
        --checkpoint-dir data/checkpoints \\
        --corpus data/corpora/tiny_shakespeare.txt \\
        --n-eval-chars 5000 \\
        --out evals/results/l1_all.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from evals.tier2_llm_parity.tiny_shakespeare import evaluate


def main() -> int:
    p = argparse.ArgumentParser(description="L1 across all checkpoints")
    p.add_argument("--checkpoint-dir", default="data/checkpoints")
    p.add_argument("--corpus", default="data/corpora/tiny_shakespeare.txt")
    p.add_argument("--n-eval-chars", type=int, default=2000)
    p.add_argument("--out", default="evals/results/l1_all.json")
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        print(f"checkpoint dir not found: {ckpt_dir}")
        return 2
    ckpts = sorted(ckpt_dir.glob("*.npz"))
    if not ckpts:
        print(f"no .npz checkpoints under {ckpt_dir}")
        return 2

    rows: list[dict] = []
    print(f"{'checkpoint':50s}  {'loss':8s}  {'carry':5s}  {'val':>8s}  {'metric':12s}")
    print("-" * 95)
    for ckpt in ckpts:
        try:
            result = evaluate(
                str(ckpt), args.corpus,
                n_eval_chars=args.n_eval_chars,
                metric="auto" if False else "nll",  # nll metric for cross-checkpoint comparison
            )
        except Exception as exc:  # noqa: BLE001 -- want to surface every error
            print(f"{ckpt.name:50s}  ERROR  {exc}")
            rows.append({"checkpoint": ckpt.name, "error": repr(exc)})
            continue
        meta = result["checkpoint_metadata"]
        loss_type = meta.get("loss_type", "mse")
        carry = meta.get("carry_steps_trained", 0)
        val = result.get("nll_loss", result.get("hv_mse_loss"))
        metric = result["metric_type"]
        print(
            f"{ckpt.name:50s}  {loss_type:8s}  c={carry:2d}  "
            f"{val:8.4f}  {metric:12s}"
        )
        rows.append({"checkpoint": ckpt.name, "result": result})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nfull report written to {out_path}")
    return 0


# Patch: use the same metric for all checkpoints to make the table sortable.
# We use NLL across the board; MSE-trained checkpoints get a meaningless
# (high) NLL because their HYMN output isn't softmax-aligned, but that's
# the honest comparison vs the NLL-trained checkpoints.

if __name__ == "__main__":
    raise SystemExit(main())
