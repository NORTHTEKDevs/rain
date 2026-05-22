# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HYMN bootstrap pre-training driver.

Phase 1.3 of the training protocol. This is the ONLY place backprop appears
in RAIN. After this script runs, HYMN weights are frozen and Phase 2 uses
local-rule learning only.

Scope of v0 implementation: pure numpy + differentiable tanh surrogate of
the bipolar sign() activation in the actual Rust HYMN. Trained weights port
to the Rust model via Task 2.4's checkpoint format.
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from rain.core.relational import Codebook
from rain.train.checkpoint import HymnCheckpointMetadata, save_checkpoint, freeze_checkpoint


class HymnSurrogate:
    """Differentiable surrogate for HYMN forward, used during Phase 1 backprop.

    Identical architecture to rain-rs/src/hymn.rs (2-layer MLP) but with tanh
    activation and float32 throughout so we can backprop. Weights are exported
    to the Rust HYMN at end-of-train via Task 2.4's checkpoint format.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.W1 = rng.standard_normal((in_dim, hidden_dim)).astype(np.float32) / np.sqrt(in_dim)
        self.W2 = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32) / np.sqrt(hidden_dim)

    def forward(self, state: np.ndarray, input_: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward returning (output, cache_for_backward)."""
        combined = np.tanh(state + input_)  # bipolar-ish combine
        h_pre = combined @ self.W1
        h = np.tanh(h_pre)
        out_pre = h @ self.W2
        out = np.tanh(out_pre)
        cache = {"combined": combined, "h_pre": h_pre, "h": h, "out_pre": out_pre, "out": out}
        return out, cache

    def backward(self, d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """Returns (dW1, dW2) for a single example."""
        d_out_pre = d_out * (1 - cache["out"] ** 2)
        dW2 = cache["h"][:, None] @ d_out_pre[None, :]
        d_h = d_out_pre @ self.W2.T
        d_h_pre = d_h * (1 - cache["h"] ** 2)
        dW1 = cache["combined"][:, None] @ d_h_pre[None, :]
        return dW1, dW2

    def step(self, dW1: np.ndarray, dW2: np.ndarray, lr: float) -> None:
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2


def train_steps(model, codebook, corpus_text: str, n_steps: int, lr: float = 0.001, seed: int = 0):
    """Run n_steps of next-char prediction on the corpus. Returns list of step losses."""
    rng = np.random.default_rng(seed)
    chars = list(corpus_text)
    char_set = sorted(set(chars))

    # Ensure codebook has vectors for every char in this corpus
    for c in char_set:
        codebook.vector(c)

    losses = []
    for step in range(n_steps):
        # Sample a random position
        if len(chars) < 2:
            break
        pos = rng.integers(0, len(chars) - 1)
        prev_char = chars[pos]
        next_char = chars[pos + 1]

        state = codebook.vector(prev_char).astype(np.float32)
        input_ = np.zeros_like(state)  # no prior context for char-level smoke test

        out, cache = model.forward(state, input_)

        target = codebook.vector(next_char).astype(np.float32)
        # MSE-ish loss against target hypervector
        d_out = 2 * (out - target) / model.out_dim
        loss = float(np.mean((out - target) ** 2))
        losses.append(loss)

        dW1, dW2 = model.backward(d_out, cache)
        model.step(dW1, dW2, lr)

    return losses


def main():
    parser = argparse.ArgumentParser(description="HYMN bootstrap pre-training driver")
    parser.add_argument("--corpus", type=str, required=True, help="path to text corpus")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--in-dim", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="hymn_checkpoint.npz",
                        help="output checkpoint path (Task 2.4 format)")
    parser.add_argument("--save-frozen", action="store_true",
                        help="Set frozen=True after saving (post-Phase-1 default).")
    args = parser.parse_args()

    corpus = Path(args.corpus).read_text()
    cb = Codebook(vocab_size=256, dim=args.in_dim, seed=args.seed)
    model = HymnSurrogate(args.in_dim, args.hidden_dim, args.out_dim, seed=args.seed)

    losses = train_steps(model, cb, corpus, n_steps=args.steps, lr=args.lr, seed=args.seed)

    metadata = HymnCheckpointMetadata(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        final_loss=losses[-1] if losses else None,
        initial_loss=losses[0] if losses else None,
    )
    npz_path, json_path = save_checkpoint(
        args.out, model.W1, model.W2, metadata,
        losses=np.array(losses) if losses else None,
    )
    if args.save_frozen:
        metadata = freeze_checkpoint(args.out)

    print(f"Saved checkpoint to {npz_path} + {json_path}")
    print(f"Frozen: {metadata.frozen}")
    print(f"Initial loss: {metadata.initial_loss}, Final loss: {metadata.final_loss}")


if __name__ == "__main__":
    main()
