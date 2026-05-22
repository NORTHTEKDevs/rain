"""Compositional reasoning via VSA binding.

The killer demo of why HV-based systems can do something transformers
structurally cannot.

We give RCK three orthogonal slot vocabularies (colour, shape, quantity)
and a tiny set of base facts.  At query time we ask for combinations
NEVER SEEN AT TRAINING:

    Training:
      red ball, blue ball, green ball
      red cube, blue cube
      one ball, two balls, three balls
      one cube
    Query (unseen during training):
      "two red cubes"          -> should compose correctly
      "three blue cubes"       -> should compose correctly

A char-level next-token model has no representation of "two red cubes"
because that sequence never appeared.  An LLM trained on the same text
will assign near-zero probability to the composition.  RCK, in contrast,
treats colour / shape / quantity as orthogonal slots and BINDS them on
the fly -- the answer comes from the algebra of hypervectors, not from
memorised n-grams.

This module also provides federated bundling utilities and editable
codebook surgery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np

from rck.codebook import Codebook
from rck.relational import RelationalMemory
from rck.vsa import bind


@dataclass
class CompositionalReasoner:
    """Slot-based compositional reasoner.

    Each `slot` is a role (e.g. "color", "shape", "quantity"). A fact is
    stored as the multiplicative bind of role-tagged filler values plus
    a binding of role_OUT to the rendered output. Query: given known
    slots, recover the OUT.
    """

    dim: int = 4096
    seed: int = 0
    out_role: str = "OUT"
    codebook: Codebook = field(default=None, init=False)
    memory: RelationalMemory = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self.memory = RelationalMemory(
            dim=self.dim, seed=self.seed,
            role_names=("color", "shape", "quantity", self.out_role),
        )

    # ---- training ----------------------------------------------------------

    def teach(self, slots: dict[str, Hashable], output: Hashable) -> None:
        """Store one labelled example: a slot assignment -> output."""
        fact = dict(slots)
        fact[self.out_role] = output
        self.memory.store(self.codebook, fact)

    def teach_pair(self, slot: str, value: Hashable, output: Hashable) -> None:
        """One-slot primitive: 'red' -> 'r', 'ball' -> 'B', etc."""
        self.memory.store(self.codebook, {slot: value, self.out_role: output})

    # ---- inference ---------------------------------------------------------

    def answer(self, slots: dict[str, Hashable], top_k: int = 1) -> list[tuple[Hashable, float]]:
        """Return top-k candidate outputs for the given slot assignment."""
        return self.memory.query(self.codebook, slots, self.out_role, top_k=top_k)

    def compose(self, slots: dict[str, Hashable]) -> tuple[Hashable, str, float]:
        """Composition by ALGEBRAIC SUM of single-slot answers.

        For each slot in `slots`, query the unknown OUT given just that
        slot -- this returns the slot's PRIMITIVE output (e.g. red -> 'r').
        Then concatenate or bundle.

        Returns (composed_symbol_or_None, render_string, mean_score).
        """
        parts: list[str] = []
        scores: list[float] = []
        for slot, val in slots.items():
            ans, score = self._single_slot_answer(slot, val)
            if ans is not None:
                parts.append(str(ans))
                scores.append(score)
        return None, " ".join(parts), float(np.mean(scores)) if scores else 0.0

    def _single_slot_answer(self, slot: str, value: Hashable) -> tuple[Hashable | None, float]:
        results = self.memory.query(self.codebook, {slot: value}, self.out_role, top_k=1)
        if not results:
            return None, 0.0
        return results[0]

    # ---- federated bundling ------------------------------------------------

    def merge(self, other: "CompositionalReasoner") -> None:
        """Combine knowledge from `other` into self.

        Bundles the relational memories and unions the codebooks. The two
        agents must share `dim` and `seed` so their codebook atoms and
        role HVs match.
        """
        if other.dim != self.dim:
            raise ValueError("dim mismatch")
        if other.seed != self.seed:
            raise ValueError("seed mismatch -- codebooks must align")
        for sym, hv in other.codebook._atoms.items():
            if sym not in self.codebook._atoms:
                self.codebook._atoms[sym] = hv.copy()
        self.codebook._cache_dirty = True
        self.memory.merge(other.memory)

    # ---- editable surgery --------------------------------------------------

    def rename(self, old: Hashable, new: Hashable) -> None:
        """Swap a codebook atom under a new name. The HV stays the same so
        all facts about `old` are now retrievable as `new`."""
        if old not in self.codebook._atoms:
            return
        self.codebook._atoms[new] = self.codebook._atoms.pop(old)
        # Patch fact list.
        for f in self.memory._facts:
            for k, v in list(f.items()):
                if v == old:
                    f[k] = new
        self.codebook._cache_dirty = True

    def forget_fact(self, slots: dict[str, Hashable], output: Hashable) -> None:
        fact = dict(slots)
        fact[self.out_role] = output
        self.memory.forget(self.codebook, fact)
