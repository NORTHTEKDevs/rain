"""VSA n-gram memory -- pure-symbol next-token prediction.

This is the classic Kanerva HRR n-gram trick: encode each (context, next)
pair as a bound hypervector, bundle them into a single memory vector, then
query by unbinding the context at inference time.

For an n-gram of order k:
    role(i)   = permute(codebook[x_{t-i}], shift=i+1)        for i = 0..k-1
    key       = bind(role(k-1), bind(role(k-2), ... role(0)))
    memory    += bind(key, codebook[x_{t+1}])

At decode time:
    query     = unbind(memory, key)
    next      = cleanup(query, codebook)

The memory is the dim-D real-valued accumulation -- we never sign() it so
that the relative magnitude of next-char votes survives.

This module sits ALONGSIDE the FEP, not inside it. The agent collects both
the LSM-readout candidate and the n-gram candidate, then EFE breaks ties.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from rck.codebook import Codebook
from rck.vsa import bind, permute


@dataclass
class BigramMemory:
    """VSA n-gram associative memory."""

    dim: int = 4096
    order: int = 2  # how many prior chars to use as context
    decay: float = 1.0  # if <1, older bindings decay each store (forgetting)
    _mem: np.ndarray = field(default=None, init=False)
    _ctx: list[Hashable] = field(default_factory=list, init=False)
    _next_counts: Counter = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        self._mem = np.zeros(self.dim, dtype=np.float32)

    # ---- key construction --------------------------------------------------

    def _role(self, codebook: Codebook, symbol: Hashable, position: int) -> np.ndarray:
        return permute(codebook.encode(symbol), position + 1)

    def _key(self, codebook: Codebook, context: list[Hashable]) -> np.ndarray:
        """Bind the role-tagged HVs of the (up to `order`) prior symbols."""
        ctx = context[-self.order:]
        if not ctx:
            # Empty context -> all-ones key (acts as a global unigram bucket).
            return np.ones(self.dim, dtype=np.int8)
        key = self._role(codebook, ctx[0], 0)
        for i, sym in enumerate(ctx[1:], start=1):
            key = bind(key, self._role(codebook, sym, i))
        return key

    # ---- core ops ----------------------------------------------------------

    def observe(
        self,
        codebook: Codebook,
        symbol: Hashable,
        next_symbol: Hashable | None,
    ) -> None:
        """Append `symbol` to the rolling context. If next_symbol is given,
        store the (context, next_symbol) pair into memory."""
        self._ctx.append(symbol)
        if len(self._ctx) > self.order:
            self._ctx = self._ctx[-self.order:]
        if next_symbol is None:
            return
        key = self._key(codebook, self._ctx)
        binding = bind(key, codebook.encode(next_symbol))
        if self.decay < 1.0:
            self._mem *= self.decay
        self._mem += binding.astype(np.float32)
        self._next_counts[next_symbol] += 1

    def query(
        self,
        codebook: Codebook,
        context: list[Hashable] | None = None,
        top_k: int = 5,
    ) -> list[tuple[Hashable, float]]:
        """Return top-k most likely next symbols given the context."""
        ctx = context if context is not None else self._ctx
        key = self._key(codebook, ctx)
        # Unbind: with bipolar key, bind(key, bind(key, x)) = x. We're querying
        # over a real-valued mem, so the same trick applies: key * mem returns
        # the bundled set of next-symbols.
        query_hv = key.astype(np.float32) * self._mem
        return codebook.fast_cleanup(query_hv, top_k=top_k)

    def reset_context(self) -> None:
        self._ctx = []

    def unigram_top(self, k: int = 5) -> list[Hashable]:
        return [s for s, _ in self._next_counts.most_common(k)]

    def size(self) -> int:
        """Total bindings stored (sum of counts)."""
        return sum(self._next_counts.values())
