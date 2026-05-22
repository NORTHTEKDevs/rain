# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/relational.py — adapted from 4K FHRR to 10K bipolar
"""Bipolar 10K-dim VSA primitives. Bind = element-wise product. Bundle = sum + sign."""

from __future__ import annotations

import hashlib
import numpy as np

DEFAULT_DIM = 10000


def _seed_hash(name: str, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


class Codebook:
    """Deterministic hypervector codebook over a vocabulary."""

    def __init__(self, vocab_size: int, dim: int = DEFAULT_DIM, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}

    def vector(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        rng = np.random.default_rng(_seed_hash(name, self.seed))
        v = rng.choice(np.array([-1, 1], dtype=np.int16), size=self.dim)
        self._cache[name] = v
        return v


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bipolar bind = element-wise product. Self-inverse."""
    return (a * b).astype(np.int16)


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Bipolar unbind = bind (because product is self-inverse over {-1, +1})."""
    return bind(bound, key)


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Bipolar bundle = element-wise sum then sign."""
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.sign(summed + (summed == 0)).astype(np.int16)  # break ties to +1
