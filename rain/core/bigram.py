# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/bigram.py — adapted from RCK FHRR.
"""VSA n-gram memory. EFE decoder signal #3.

Stores (context_n -> next_token) associations via HRR binding. Cleanup via
codebook nearest-neighbor lookup. Vectorized for fast batch query.
"""

from __future__ import annotations
import numpy as np
from rain.core.relational import Codebook, bind, bundle


class BigramMemory:
    def __init__(self, codebook: Codebook, order: int = 2) -> None:
        self.codebook = codebook
        self.order = order
        self.D = codebook.dim
        # Memory hypervector: sum of (context-bound -> next-token) pairs
        self.memory: np.ndarray = np.zeros(self.D, dtype=np.int32)
        self.count: int = 0

    def _context_hv(self, context_tokens: list[str]) -> np.ndarray:
        """Bind the last `order` tokens into one context hypervector."""
        if not context_tokens:
            raise ValueError("context_tokens must be non-empty")
        ctx_tokens = context_tokens[-self.order:]
        # Bind sequentially: a, b, c -> bind(a, bind(b, c))
        result = self.codebook.vector(ctx_tokens[0])
        for t in ctx_tokens[1:]:
            result = bind(result, self.codebook.vector(t))
        return result

    def add(self, context_tokens: list[str], next_token: str) -> None:
        ctx = self._context_hv(context_tokens)
        nxt = self.codebook.vector(next_token)
        # Bind context -> next as one HV, add to memory bundle
        contribution = bind(ctx, nxt).astype(np.int32)
        self.memory += contribution
        self.count += 1

    def query(self, context_tokens: list[str], candidate_tokens: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        """Given a context, return top-k candidates ranked by cleanup similarity."""
        if self.count == 0:
            return []
        ctx = self._context_hv(context_tokens)
        # Unbind: bound_next ~= bind(ctx, memory)
        bound_next = bind(ctx, np.sign(self.memory + (self.memory == 0)).astype(np.int8))
        # Compare to each candidate
        scored = []
        for tok in candidate_tokens:
            cand_hv = self.codebook.vector(tok)
            sim = float(np.dot(bound_next.astype(np.float32), cand_hv.astype(np.float32)) / self.D)
            scored.append((tok, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
