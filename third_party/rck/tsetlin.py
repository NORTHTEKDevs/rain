"""Tsetlin Machine layer for interpretable causal inference.

Each clause is a conjunction of literals over a binary feature vector. Half
the clauses vote for a class, half against. Literals are toggled in/out by
Tsetlin automata (finite-state machines) that respond to Type-I (boost true
positives, fix false negatives) and Type-II (penalize false positives)
feedback.

Reference: Granmo 2018, "The Tsetlin Machine -- A Game Theoretic Bandit
Driven Approach to Optimal Pattern Recognition with Propositional Logic."

v1.0: fully vectorized evaluate + feedback. The per-clause Python loops in
v0.x are replaced with matrix ops over the (n_clauses, 2*n_features) TA
state and a pre-sampled random matrix. Roughly 10x faster at the
n_clauses=32, D=2048 scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TsetlinLayer:
    """Single-output Tsetlin Machine with positive and negative clauses.

    Args:
        n_features: input feature width (we'll feed the workspace HV, dim D).
        n_clauses: total clauses; half positive, half negative.
        T: voting threshold (controls update saturation).
        s: specificity (controls literal inclusion probability during type-I).
        n_states: states per Tsetlin automaton (must be even).
        seed: rng seed.
    """

    n_features: int
    n_clauses: int = 64
    T: float = 8.0
    s: float = 3.9
    n_states: int = 100
    seed: int = 0

    _ta: np.ndarray = field(default=None, init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even")
        if self.n_states % 2 != 0:
            raise ValueError("n_states must be even")
        self._rng = np.random.default_rng(self.seed)
        self._ta = np.full(
            (self.n_clauses, 2 * self.n_features),
            self.n_states // 2,
            dtype=np.int32,
        )

    # ---- helpers -----------------------------------------------------------

    def _include_mask(self) -> np.ndarray:
        return self._ta > (self.n_states // 2)

    @staticmethod
    def _binarize(hv: np.ndarray) -> np.ndarray:
        return (hv > 0).astype(np.int8)

    def _literals(self, feat: np.ndarray) -> np.ndarray:
        return np.concatenate([feat, 1 - feat]).astype(np.int8)

    # ---- vectorized ops ----------------------------------------------------

    def evaluate(self, hv: np.ndarray) -> tuple[float, np.ndarray]:
        """clause_out[j] = 1 iff every included literal has value 1.

        Vectorized as: violation[j,k] = inc[j,k] AND (lits[k] == 0).
        clause_out[j] = 1 - any(violation[j, :]).
        Vacuous clauses (no included literals) output 1.
        """
        feat = self._binarize(hv)
        lits = self._literals(feat)
        inc = self._include_mask()
        violation = inc & (lits == 0)[None, :]
        clause_out = (~violation.any(axis=1)).astype(np.int8)
        half = self.n_clauses // 2
        score = float(clause_out[:half].sum() - clause_out[half:].sum())
        return score, clause_out

    def feedback(self, hv: np.ndarray, target: int) -> None:
        """Vectorized Tsetlin feedback for target in {-1, +1}."""
        if target not in (-1, 1):
            raise ValueError("target must be -1 or +1")
        score, clause_out = self.evaluate(hv)
        score_clip = max(-self.T, min(self.T, score))
        p_update = (self.T - target * score_clip) / (2 * self.T)

        # Per-clause: receive feedback at all? (binary mask)
        upd_mask = self._rng.random(self.n_clauses) < p_update
        if not upd_mask.any():
            return

        feat = self._binarize(hv)
        lits = self._literals(feat).astype(np.int8)
        L = lits.shape[0]
        half = self.n_clauses // 2

        # Polarity of each clause (+1 for first half, -1 for second).
        polarity = np.where(np.arange(self.n_clauses) < half, 1, -1)
        # Type-I if polarity == target, else Type-II.
        type_i_mask = upd_mask & (polarity == target)
        type_ii_mask = upd_mask & (polarity != target)

        # Pre-sample a single random matrix; reuse for both types.
        rand = self._rng.random((self.n_clauses, L)).astype(np.float32)
        s_minus_one_over_s = (self.s - 1) / self.s
        one_over_s = 1.0 / self.s

        # ----- Type-I -------------------------------------------------------
        if type_i_mask.any():
            # For each clause j in type_i:
            #   if clause_out[j] == 1 (fired): boost true literals (lits==1),
            #     forget false ones (lits==0).
            #   if clause_out[j] == 0 (didn't fire): forget randomly (prob 1/s).
            fired = clause_out.astype(bool)  # (n_clauses,)
            rows = type_i_mask  # (n_clauses,)

            # Clauses that fired: boost matching literals, forget mismatches.
            fired_rows = rows & fired
            if fired_rows.any():
                boost_mask = fired_rows[:, None] & (lits == 1)[None, :] & (rand < s_minus_one_over_s)
                forget_mask = fired_rows[:, None] & (lits == 0)[None, :] & (rand < one_over_s)
                self._ta[boost_mask] = np.minimum(self._ta[boost_mask] + 1, self.n_states - 1)
                self._ta[forget_mask] = np.maximum(self._ta[forget_mask] - 1, 0)

            # Clauses that didn't fire: forget everything randomly.
            miss_rows = rows & ~fired
            if miss_rows.any():
                forget_mask = miss_rows[:, None] & (rand < one_over_s)
                self._ta[forget_mask] = np.maximum(self._ta[forget_mask] - 1, 0)

        # ----- Type-II ------------------------------------------------------
        if type_ii_mask.any():
            # Force exclusion of literals that made a false positive fire:
            #   for clauses that fired (clause_out==1), find literals with
            #   value 0 that are not yet included, and INCREMENT them so they
            #   become included on the next round.
            inc = self._include_mask()
            fired_rows = type_ii_mask & clause_out.astype(bool)
            if fired_rows.any():
                bump_mask = fired_rows[:, None] & (lits == 0)[None, :] & (~inc)
                self._ta[bump_mask] = np.minimum(self._ta[bump_mask] + 1, self.n_states - 1)

    # ---- explanation -------------------------------------------------------

    def explain(self, hv: np.ndarray, feature_names: list[str] | None = None,
                max_clauses: int = 8) -> list[str]:
        """Human-readable clauses that FIRED on this input."""
        feat = self._binarize(hv)
        lits = self._literals(feat)
        inc = self._include_mask()
        # Find firing clauses (no included literal violated).
        violation = inc & (lits == 0)[None, :]
        firing = ~violation.any(axis=1) & inc.any(axis=1)  # exclude vacuous
        names = feature_names or [f"f{i}" for i in range(self.n_features)]
        half = self.n_clauses // 2
        out = []
        for j in np.where(firing)[0][:max_clauses]:
            parts = []
            for k in np.where(inc[j])[0]:
                if k < self.n_features:
                    parts.append(names[k])
                else:
                    parts.append(f"NOT {names[k - self.n_features]}")
            polarity = "+" if j < half else "-"
            out.append(f"({polarity}) " + " AND ".join(parts))
        return out
