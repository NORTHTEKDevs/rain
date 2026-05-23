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
                        top_k: int | None = None,
                        recent_ids: list[int] | None = None,
                        repetition_penalty: float = 1.0) -> int:
    """Numerically-stable softmax sampling. temperature<=0 -> argmax.

    repetition_penalty > 1.0 down-weights any vocab id present in
    `recent_ids` (the model's last few outputs), to prevent the argmax
    mode-collapse loop ('a promised a promised a promised...').
    """
    scaled_logits = logits.astype(np.float64).copy()
    if recent_ids and repetition_penalty > 1.0:
        for idx in set(recent_ids):
            if 0 <= idx < len(scaled_logits):
                # Discount positive logits, amplify negative ones -- standard
                # HuggingFace-style repetition penalty.
                if scaled_logits[idx] > 0:
                    scaled_logits[idx] /= repetition_penalty
                else:
                    scaled_logits[idx] *= repetition_penalty
    if temperature <= 0.0:
        return int(np.argmax(scaled_logits))
    scaled = scaled_logits / temperature
    m = float(scaled.max())
    probs = np.exp(scaled - m)
    if top_k is not None and top_k > 0 and top_k < len(probs):
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
    repetition_penalty: float = 1.0,
    repetition_window: int = 16,
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
    recent_ids: list[int] = []
    for _ in range(n_tokens):
        logits = codebook_matrix @ state
        next_idx = _sample_from_logits(
            logits, temperature, rng, top_k=top_k,
            recent_ids=recent_ids,
            repetition_penalty=repetition_penalty,
        )
        next_char = vocab[next_idx]
        generated_chars.append(next_char)
        recent_ids.append(next_idx)
        if len(recent_ids) > repetition_window:
            recent_ids = recent_ids[-repetition_window:]
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
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 disables; >1.0 down-weights tokens in the recent window")
    p.add_argument("--repetition-window", type=int, default=16,
                   help="how many recent tokens to consider for repetition penalty")
    args = p.parse_args()

    out = sample(
        args.checkpoint, args.corpus,
        prompt=args.prompt, n_tokens=args.n_tokens,
        temperature=args.temperature,
        top_k=(args.top_k if args.top_k > 0 else None),
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    # Force utf-8 on stdout so unicode chars in larger corpora (WikiText-2
    # has 1013 unique chars) don't crash on Windows' cp1252 default.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover -- old python
        pass
    if args.prompt:
        sys.stdout.write(args.prompt)
    sys.stdout.write(out)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
