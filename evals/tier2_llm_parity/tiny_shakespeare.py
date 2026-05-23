# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tier 2 L1 benchmark -- Tiny Shakespeare val-loss.

Two metric paths:
- `hv_mse`: MSE on bipolar HV targets. v0 proxy threshold 0.5. Use when the
  checkpoint was trained with the MSE loss path (the original numpy reference).
- `nll`: real cross-entropy nats/char via codebook-softmax projection. The
  design-plan threshold of 1.55 nats/char applies here. Use when the checkpoint
  was trained with the NLL loss path (Phase 2.3+).

The model is loaded via rain.train.checkpoint.load_checkpoint(). For a frozen
HYMN, the surrogate's tanh-float forward is what we evaluate (the Rust bipolar
forward is for inference; surrogate is for training + eval continuity).

Usage:
    # HV-MSE eval (default; matches the v0 reference)
    python evals/tier2_llm_parity/tiny_shakespeare.py \
        --checkpoint <hymn.npz> --corpus <tiny.txt> --out <out.json>

    # NLL eval (Phase 2.3 production threshold)
    python evals/tier2_llm_parity/tiny_shakespeare.py \
        --checkpoint <hymn.npz> --corpus <tiny.txt> --metric nll --out <out.json>
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from rain.core.relational import Codebook
from rain.train.checkpoint import load_checkpoint


# HV-MSE threshold calibrated for the HYMN surrogate's hypervector MSE output.
L1_PASS_THRESHOLD_HV_MSE = 0.5
# Real cross-entropy threshold from the design plan (nats per character).
L1_PASS_THRESHOLD_NLL = 1.55


def compute_hv_mse_loss(W1: np.ndarray, W2: np.ndarray, codebook: Codebook,
                     corpus_text: str, n_eval_chars: int) -> float:
    """Compute mean-squared-error on next-char prediction over n_eval_chars random positions.

    NOTE: this is MSE on hypervector predictions, NOT cross-entropy in nats/char.
    The plan's pass threshold of 1.55 nats/char corresponds to a transformer
    cross-entropy benchmark; the HYMN surrogate's MSE-on-HV is the v0 proxy.
    Empirical mapping between the two will be calibrated when real training
    lands in Phase 2.3 -- for now we return MSE under the same threshold name.
    """
    chars = list(corpus_text)
    if len(chars) < n_eval_chars + 1:
        n_eval_chars = max(1, len(chars) - 1)
    rng = np.random.default_rng(42)
    losses: list[float] = []
    for _ in range(n_eval_chars):
        pos = int(rng.integers(0, len(chars) - 1))
        prev_char = chars[pos]
        next_char = chars[pos + 1]
        state = codebook.vector(prev_char).astype(np.float32)
        input_ = np.zeros_like(state)
        combined = np.tanh(state + input_)
        h = np.tanh(combined @ W1)
        out = np.tanh(h @ W2)
        target = codebook.vector(next_char).astype(np.float32)
        losses.append(float(np.mean((out - target) ** 2)))
    return float(np.mean(losses))


def compute_nll_loss(
    W1: np.ndarray, W2: np.ndarray, codebook: Codebook,
    corpus_text: str, n_eval_chars: int,
) -> float:
    """Mean cross-entropy nats/char over n_eval_chars random positions.

    HYMN output is projected onto the codebook vocab (every char that appears
    in the corpus, in sorted order) via dot product, softmaxed, and the
    negative log-prob of the true next char is the per-position loss.
    """
    chars = list(corpus_text)
    vocab = sorted(set(chars))
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    codebook_matrix = np.stack(
        [codebook.vector(c).astype(np.float32) for c in vocab]
    )  # (V, D)
    if len(chars) < n_eval_chars + 1:
        n_eval_chars = max(1, len(chars) - 1)
    rng = np.random.default_rng(42)
    nll_total = 0.0
    for _ in range(n_eval_chars):
        pos = int(rng.integers(0, len(chars) - 1))
        prev_char = chars[pos]
        next_char = chars[pos + 1]
        state = codebook.vector(prev_char).astype(np.float32)
        input_ = np.zeros_like(state)
        combined = np.tanh(state + input_)
        h = np.tanh(combined @ W1)
        out = np.tanh(h @ W2)
        logits = codebook_matrix @ out  # (V,)
        # log-softmax in a numerically stable way
        m = float(logits.max())
        log_sum_exp = m + float(np.log(np.exp(logits - m).sum()))
        log_probs = logits - log_sum_exp
        nll_total += -float(log_probs[char_to_idx[next_char]])
    return nll_total / n_eval_chars


def evaluate(
    checkpoint_path: str,
    corpus_path: str,
    n_eval_chars: int = 1000,
    metric: str = "hv_mse",
) -> dict:
    """Run the L1 benchmark. Returns a result dict suitable for JSON serialization.

    metric: "hv_mse" (default, matches v0 numpy reference) or "nll"
    (real cross-entropy, Phase 2.3 production threshold).
    """
    if metric not in ("hv_mse", "nll"):
        raise ValueError(f"unknown metric {metric!r}; expected 'hv_mse' or 'nll'")
    W1, W2, meta = load_checkpoint(checkpoint_path)
    corpus = Path(corpus_path).read_text()
    cb = Codebook(vocab_size=256, dim=meta.in_dim, seed=meta.seed)
    base = {
        "benchmark": "L1_tiny_shakespeare",
        "n_eval_chars": n_eval_chars,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": {
            "in_dim": meta.in_dim,
            "hidden_dim": meta.hidden_dim,
            "out_dim": meta.out_dim,
            "steps": meta.steps,
            "frozen": meta.frozen,
        },
    }
    if metric == "hv_mse":
        loss = compute_hv_mse_loss(W1, W2, cb, corpus, n_eval_chars=n_eval_chars)
        base.update({
            "metric_type": "hv_mse",
            "nll_threshold_applicable": False,
            "threshold": L1_PASS_THRESHOLD_HV_MSE,
            "hv_mse_loss": loss,
            "pass": loss <= L1_PASS_THRESHOLD_HV_MSE,
        })
    else:
        loss = compute_nll_loss(W1, W2, cb, corpus, n_eval_chars=n_eval_chars)
        base.update({
            "metric_type": "nll_nats_per_char",
            "nll_threshold_applicable": True,
            "threshold": L1_PASS_THRESHOLD_NLL,
            "nll_loss": loss,
            "pass": loss <= L1_PASS_THRESHOLD_NLL,
        })
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2 L1 -- Tiny Shakespeare val-loss")
    parser.add_argument("--checkpoint", required=True, help="path to HYMN .npz checkpoint")
    parser.add_argument("--corpus", required=True, help="path to Tiny Shakespeare .txt")
    parser.add_argument("--n-eval-chars", type=int, default=1000)
    parser.add_argument("--metric", choices=["hv_mse", "nll"], default="hv_mse",
                        help="hv_mse = v0 reference; nll = real cross-entropy (1.55 nats/char target)")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.corpus,
                      n_eval_chars=args.n_eval_chars, metric=args.metric)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
