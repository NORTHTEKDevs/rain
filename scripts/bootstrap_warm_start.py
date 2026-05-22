# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Warm-start the codebook from pretrained embeddings via sign projection.

Phase 1.1 of the training protocol. Not an LLM substrate at inference -
just smart init."""

from __future__ import annotations
import numpy as np
from rain.core.relational import Codebook


def warm_start_from_vectors(cb: Codebook, vectors: dict[str, np.ndarray]) -> int:
    """Tile each input vector to reach `cb.dim`, take sign, store. Returns count loaded."""
    loaded = 0
    for name, v in vectors.items():
        repeats = (cb.dim + len(v) - 1) // len(v)
        tiled = np.tile(v, repeats)[: cb.dim]
        bipolar = np.sign(tiled + (tiled == 0)).astype(np.int16)
        cb._cache[name] = bipolar
        loaded += 1
    return loaded
