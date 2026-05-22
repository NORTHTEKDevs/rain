# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/tsetlin.py -- adapted from RCK 4K FHRR to
# bipolar 10K-dim and simplified for v0 RAIN.
"""Vectorised Tsetlin Machine for bipolar hypervector features.

Each clause is a vector of inclusion bits: clause_inclusion[c, f] in {0, 1}
indicates whether feature f is included in clause c (positive literal). A
clause "fires" on a state if all its included features are +1 in the state.

For v0 simplicity we do NOT implement the negative-literal (NOT) inclusions
that the full Tsetlin Machine supports; clauses are conjunctions of positive
literals only. Adding negative literals is a v0.5 refinement.

Voting: each clause is polarity-assigned (positive or negative) per class.
Per-class vote = sum of firing positive-clauses - sum of firing negative-clauses.

Feedback: a simple Type-I rule -- when target_class != predicted_class, randomly
include a feature that fires in the target state OR exclude a feature that
fires in the predicted state, for clauses of that polarity.
"""

from __future__ import annotations
import numpy as np


class TsetlinMachine:
    def __init__(
        self,
        num_classes: int,
        num_clauses_per_class: int,
        num_features: int,
        threshold_T: int = 10,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.num_clauses_per_class = num_clauses_per_class
        self.num_features = num_features
        self.T = threshold_T
        self.rng = np.random.default_rng(seed)

        # Clause inclusion bits per class. Shape: (num_classes, num_clauses_per_class, num_features)
        # Each clause's included features are sparse; initialize empty (no features included).
        self.inclusion = np.zeros(
            (num_classes, num_clauses_per_class, num_features), dtype=np.int8
        )
        # Clause polarity per class: half positive, half negative.
        # Shape: (num_classes, num_clauses_per_class) -- +1 or -1.
        polarities = np.array(
            [1 if i % 2 == 0 else -1 for i in range(num_clauses_per_class)], dtype=np.int8
        )
        self.polarity = np.tile(polarities, (num_classes, 1))

    def _clause_fires(self, state: np.ndarray, inclusion_row: np.ndarray) -> bool:
        """A clause fires if all its INCLUDED features are +1 in the state.

        With no features included (initial state), a clause fires trivially --
        but trivial fires don't contribute to discriminative power; we treat
        empty clauses as not firing to avoid noise.
        """
        if not inclusion_row.any():
            return False
        # included features must all be +1 in state
        relevant = state[inclusion_row.astype(bool)]
        return bool(np.all(relevant == 1))

    def vote(self, state: np.ndarray) -> np.ndarray:
        """Return per-class scores. Shape: (num_classes,)."""
        if state.shape != (self.num_features,):
            raise ValueError(
                f"state shape {state.shape} != ({self.num_features},)"
            )
        scores = np.zeros(self.num_classes, dtype=np.float32)
        for cls in range(self.num_classes):
            for c in range(self.num_clauses_per_class):
                if self._clause_fires(state, self.inclusion[cls, c]):
                    scores[cls] += float(self.polarity[cls, c])
        # Threshold normalization: clip to +/- T
        return np.clip(scores, -self.T, self.T)

    def predict(self, state: np.ndarray) -> int:
        return int(np.argmax(self.vote(state)))

    def feedback(self, state: np.ndarray, target_class: int) -> None:
        """Type-I-lite feedback: encourage target class's positive clauses to fire on this state,
        discourage other classes' positive clauses from firing on this state."""
        if target_class < 0 or target_class >= self.num_classes:
            raise ValueError(f"target_class out of range: {target_class}")

        # For positive clauses of target_class that DON'T fire on this state,
        # randomly add a feature where state is +1 (encourage firing).
        plus_state_idx = np.flatnonzero(state == 1)
        for c in range(self.num_clauses_per_class):
            if self.polarity[target_class, c] != 1:
                continue
            if self._clause_fires(state, self.inclusion[target_class, c]):
                continue
            if len(plus_state_idx) > 0:
                feat = int(self.rng.choice(plus_state_idx))
                self.inclusion[target_class, c, feat] = 1

        # For positive clauses of OTHER classes that DO fire on this state,
        # randomly drop one included feature (discourage firing).
        for cls in range(self.num_classes):
            if cls == target_class:
                continue
            for c in range(self.num_clauses_per_class):
                if self.polarity[cls, c] != 1:
                    continue
                if not self._clause_fires(state, self.inclusion[cls, c]):
                    continue
                included = np.flatnonzero(self.inclusion[cls, c])
                if len(included) > 0:
                    feat = int(self.rng.choice(included))
                    self.inclusion[cls, c, feat] = 0
