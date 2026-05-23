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
    carry_steps: int = 0,
) -> float:
    """Mean cross-entropy nats/char over n_eval_chars random positions.

    HYMN output is projected onto the codebook vocab (every char that appears
    in the corpus, in sorted order) via dot product, softmaxed, and the
    negative log-prob of the true next char is the per-position loss.

    If `carry_steps > 0`, evaluates RNN-style with teacher forcing across a
    `carry_steps + 1` window per sample (matches the carry-mode trainer):
    state starts at zero, ingests each char in turn via HYMN, and the
    final prediction is the target.
    """
    chars = list(corpus_text)
    vocab = sorted(set(chars))
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    codebook_matrix = np.stack(
        [codebook.vector(c).astype(np.float32) for c in vocab]
    )  # (V, D)
    if len(chars) < n_eval_chars + carry_steps + 2:
        n_eval_chars = max(1, len(chars) - carry_steps - 2)
    rng = np.random.default_rng(42)
    nll_total = 0.0

    def _hymn_forward(state: np.ndarray, input_: np.ndarray) -> np.ndarray:
        combined = np.tanh(state + input_)
        h = np.tanh(combined @ W1)
        return np.tanh(h @ W2)

    def _nll_at(state: np.ndarray, next_idx: int) -> float:
        logits = codebook_matrix @ state
        m = float(logits.max())
        log_sum_exp = m + float(np.log(np.exp(logits - m).sum()))
        return float(-(logits[next_idx] - log_sum_exp))

    in_dim = W1.shape[0]
    for _ in range(n_eval_chars):
        if carry_steps > 0:
            pos = int(rng.integers(0, len(chars) - carry_steps - 1))
            state = np.zeros(in_dim, dtype=np.float32)
            for t in range(carry_steps):
                cur_vec = codebook.vector(chars[pos + t]).astype(np.float32)
                state = _hymn_forward(state, cur_vec)
            target_idx = char_to_idx[chars[pos + carry_steps]]
            # One more transition to align with training (the trainer
            # transitions THEN decodes); to keep symmetry, do the same here.
            cur_vec = codebook.vector(chars[pos + carry_steps - 1]).astype(np.float32)
            nll_total += _nll_at(state, target_idx)
        else:
            pos = int(rng.integers(0, len(chars) - 1))
            prev_char = chars[pos]
            next_char = chars[pos + 1]
            state = codebook.vector(prev_char).astype(np.float32)
            input_ = np.zeros_like(state)
            out = _hymn_forward(state, input_)
            target_idx = char_to_idx[next_char]
            nll_total += _nll_at(out, target_idx)
    return nll_total / n_eval_chars


def evaluate(
    checkpoint_path: str,
    corpus_path: str,
    n_eval_chars: int = 1000,
    metric: str = "hv_mse",
    carry_steps: int | None = None,
) -> dict:
    """Run the L1 benchmark. Returns a result dict suitable for JSON serialization.

    metric: "hv_mse" (default, matches v0 numpy reference) or "nll"
    (real cross-entropy, Phase 2.3 production threshold).

    carry_steps: if not None, overrides the eval mode. If None, falls back
    to the checkpoint's recorded carry_steps (schema v2+). 0 = per-position
    eval; >0 = RNN-style teacher-forced eval matching the carry trainer.
    """
    if metric not in ("hv_mse", "nll"):
        raise ValueError(f"unknown metric {metric!r}; expected 'hv_mse' or 'nll'")
    W1, W2, meta = load_checkpoint(checkpoint_path)
    corpus = Path(corpus_path).read_text(encoding="utf-8")
    cb = Codebook(vocab_size=256, dim=meta.in_dim, seed=meta.seed)
    effective_carry = carry_steps if carry_steps is not None else getattr(meta, "carry_steps", 0)
    base = {
        "benchmark": "L1_tiny_shakespeare",
        "n_eval_chars": n_eval_chars,
        "eval_carry_steps": effective_carry,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": {
            "in_dim": meta.in_dim,
            "hidden_dim": meta.hidden_dim,
            "out_dim": meta.out_dim,
            "steps": meta.steps,
            "frozen": meta.frozen,
            "loss_type": getattr(meta, "loss_type", "mse"),
            "context_len": getattr(meta, "context_len", 0),
            "carry_steps_trained": getattr(meta, "carry_steps", 0),
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
        loss = compute_nll_loss(W1, W2, cb, corpus,
                                n_eval_chars=n_eval_chars,
                                carry_steps=effective_carry)
        base.update({
            "metric_type": "nll_nats_per_char",
            "nll_threshold_applicable": True,
            "threshold": L1_PASS_THRESHOLD_NLL,
            "nll_loss": loss,
            "pass": loss <= L1_PASS_THRESHOLD_NLL,
        })
    return base


def _auto_metric_from_checkpoint(checkpoint_path: str) -> str:
    """Read the checkpoint sidecar and infer the metric to use.

    If the sidecar carries `loss_type` (schema_version >= 2), match it:
    nll-trained checkpoints get nll eval, mse-trained get hv_mse eval.
    Older sidecars default to hv_mse for backward compat.
    """
    _, _, meta = load_checkpoint(checkpoint_path)
    return "nll" if meta.loss_type == "nll" else "hv_mse"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2 L1 -- Tiny Shakespeare val-loss")
    parser.add_argument("--checkpoint", required=True, help="path to HYMN .npz checkpoint")
    parser.add_argument("--corpus", required=True, help="path to Tiny Shakespeare .txt")
    parser.add_argument("--n-eval-chars", type=int, default=1000)
    parser.add_argument("--metric", choices=["hv_mse", "nll", "auto"], default="auto",
                        help="auto = pick based on the checkpoint's loss_type sidecar field "
                             "(default; matches how the checkpoint was trained). "
                             "hv_mse / nll force a specific metric.")
    parser.add_argument("--carry-steps", type=int, default=None,
                        help="Override eval mode: 0 = per-position, K>0 = RNN-style "
                             "teacher-forced K-step eval. Default = use the checkpoint's "
                             "recorded carry_steps (schema v2+) or 0 for older checkpoints.")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()
    metric = (
        _auto_metric_from_checkpoint(args.checkpoint)
        if args.metric == "auto" else args.metric
    )
    result = evaluate(args.checkpoint, args.corpus,
                      n_eval_chars=args.n_eval_chars, metric=metric,
                      carry_steps=args.carry_steps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
