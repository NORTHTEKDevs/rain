# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Sample characters from a trained HYMN checkpoint.

Autoregressive char-level generation: ingest a prompt via the HYMN
RNN-style state transition, then sample N new chars from the projected
codebook softmax. The exact same forward semantics the carry-mode
trainer uses (`scripts/pretrain_hymn_torch.py --carry-steps W`),
inverted for inference.

Usage:
    python -m scripts.sample_hymn \\
        --checkpoint data/checkpoints/hymn_carry8_30k.npz \\
        --corpus data/corpora/tiny_shakespeare.txt \\
        --prompt "ROMEO:" --n-tokens 200 --temperature 0.8

The corpus is needed only to recover the char-vocab (sorted set of chars
seen during training). All inference math is numpy -- no torch dep.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

from rain.core.relational import Codebook
from rain.train.checkpoint import load_checkpoint


def _hymn_forward(state: np.ndarray, input_: np.ndarray,
                  W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    combined = np.tanh(state + input_)
    h = np.tanh(combined @ W1)
    return np.tanh(h @ W2)


def _sample_from_logits(logits: np.ndarray, temperature: float,
                        rng: np.random.Generator,
                        top_k: int | None = None) -> int:
    """Numerically-stable softmax sampling. temperature<=0 -> argmax."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    m = float(scaled.max())
    probs = np.exp(scaled - m)
    if top_k is not None and top_k > 0 and top_k < len(probs):
        # Mask to top-k indices
        idx = np.argpartition(-probs, top_k)[:top_k]
        mask = np.full_like(probs, -np.inf)
        mask[idx] = scaled[idx]
        m2 = float(mask.max())
        probs = np.exp(mask - m2)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def sample(
    checkpoint_path: str,
    corpus_path: str,
    prompt: str = "",
    n_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    W1, W2, meta = load_checkpoint(checkpoint_path)
    corpus = Path(corpus_path).read_text(encoding="utf-8")
    vocab = sorted(set(corpus))
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    cb = Codebook(vocab_size=256, dim=meta.in_dim, seed=meta.seed)
    codebook_matrix = np.stack(
        [cb.vector(c).astype(np.float32) for c in vocab]
    )  # (V, D)

    rng = np.random.default_rng(seed)
    state = np.zeros(meta.in_dim, dtype=np.float32)

    # Ingest the prompt -- unknown chars are mapped to the first vocab char
    # (a structural fallback; should rarely matter on a well-trained corpus).
    for c in prompt:
        if c not in char_to_idx:
            c = vocab[0]
        state = _hymn_forward(state, cb.vector(c).astype(np.float32), W1, W2)

    generated_chars: list[str] = []
    for _ in range(n_tokens):
        logits = codebook_matrix @ state
        next_idx = _sample_from_logits(logits, temperature, rng, top_k=top_k)
        next_char = vocab[next_idx]
        generated_chars.append(next_char)
        state = _hymn_forward(state, codebook_matrix[next_idx], W1, W2)
    return "".join(generated_chars)


def main() -> int:
    p = argparse.ArgumentParser(description="Sample from a HYMN checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True,
                   help="path to the training corpus (for char vocab)")
    p.add_argument("--prompt", default="",
                   help="prefix to seed the state with before sampling begins")
    p.add_argument("--n-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="<=0 = greedy argmax")
    p.add_argument("--top-k", type=int, default=0,
                   help="0 = full vocab; K>0 = sample only from top-K logits")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = sample(
        args.checkpoint, args.corpus,
        prompt=args.prompt, n_tokens=args.n_tokens,
        temperature=args.temperature,
        top_k=(args.top_k if args.top_k > 0 else None),
        seed=args.seed,
    )
    if args.prompt:
        sys.stdout.write(args.prompt)
    sys.stdout.write(out)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
