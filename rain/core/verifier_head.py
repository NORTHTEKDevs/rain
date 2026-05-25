# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Verifier head: scores candidate outputs for test-time-compute sampling.

The o1 / o3 result is that test-time compute (TTC) -- sampling N candidates
and choosing the best via a verifier -- beats training compute past a
certain point. RAIN-Net uses TTC at small scale.

For each query:
    1. Sample N candidate answer HVs from the routed experts.
    2. Score each candidate via VerifierHead(query_hv, answer_hv) -> float.
    3. Pick the highest-scoring candidate (or weighted vote).
    4. The chosen candidate's score is also our confidence value.

Confidence below threshold triggers the active distillation loop (see
rain.training.active_learning): the model asks a teacher LLM for help,
ingests the teacher's answer, and updates itself for next time.

The verifier is trained on (query_hv, answer_hv, label) triples where
label comes from a judge LLM (Ollama local first, Claude API fallback).
Training data accumulates automatically as the system runs.

This module ships two implementations:
    HVVerifierHead   pure HV cosine + small learned projection. CPU-cheap.
    NeuralVerifierHead  small torch MLP. Higher capacity, needs GPU.

The HV version is used in v0.1 default. The neural version is a drop-in
upgrade when GPU is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rain.core.hv_substrate import DEFAULT_DIM, bind, similarity


# -------------------- pure-HV verifier --------------------


@dataclass
class HVVerifierHead:
    """HV-only verifier. Score = cosine(bind(query, answer), W).

    The "weight" W is a single float-valued accumulator that represents
    the learned notion of "a good (query, answer) pair." Updated online
    via additive perceptron-style steps:
        W <- W + eta * label * bind(q, a)
    Internally W is kept as float so small updates accumulate. Cosine
    similarity is scale-invariant, so we don't need to re-bipolarize.

    For exporting a bipolar version (debugging, hardware deployment),
    call .as_bipolar(). For most uses, just score() and update().
    """

    dim: int = DEFAULT_DIM
    learning_rate: float = 0.1
    _w: np.ndarray = field(init=False)
    _n_updates: int = 0

    def __post_init__(self) -> None:
        # Initialise W as a random bipolar HV (no prior assumption).
        # Internal state is float so updates accumulate.
        rng = np.random.default_rng(7)
        self._w = (rng.integers(0, 2, size=self.dim) * 2 - 1).astype(np.float32)

    def score(self, query_hv: np.ndarray, answer_hv: np.ndarray) -> float:
        """Return a confidence score in [-1, +1].

        Score > 0 = answer looks good for this query.
        Score < 0 = answer looks bad for this query.
        Magnitude = confidence.
        """
        pair = bind(query_hv, answer_hv)
        return similarity(pair, self._w)

    def update(self, query_hv: np.ndarray, answer_hv: np.ndarray, label: int) -> None:
        """Online perceptron update.

        label: +1 if answer was judged correct, -1 if judged wrong.
        """
        if label not in (-1, +1):
            raise ValueError(f"label must be +1 or -1; got {label}")
        pair = bind(query_hv, answer_hv).astype(np.float32)
        self._w = self._w + float(label) * self.learning_rate * pair
        self._n_updates += 1

    def as_bipolar(self) -> np.ndarray:
        """Return a bipolar quantized copy of W for export/inspection."""
        out = np.sign(self._w).astype(np.float32)
        out = np.where(out == 0.0, 1.0, out).astype(np.float32)
        return out

    def save(self, path) -> None:
        """Persist verifier weights + update count to disk (.npz)."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, w=self._w, n_updates=np.array(self._n_updates))

    @classmethod
    def load(cls, path, learning_rate: float = 0.1) -> "HVVerifierHead":
        """Load a trained verifier from disk. Restores W + update count.

        Returns a new HVVerifierHead with the saved state. Caller can
        keep training via .update() or just .score() at inference.
        """
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"verifier checkpoint not found: {p}")
        data = np.load(p)
        w = data["w"].astype(np.float32)
        dim = int(w.shape[0])
        v = cls(dim=dim, learning_rate=learning_rate)
        v._w = w
        v._n_updates = int(data["n_updates"]) if "n_updates" in data.files else 0
        return v

    @property
    def n_updates(self) -> int:
        return self._n_updates


# -------------------- TTC sampler --------------------


def sample_and_select(
    candidate_hvs: list[np.ndarray],
    query_hv: np.ndarray,
    verifier: HVVerifierHead,
    return_scores: bool = False,
) -> tuple[np.ndarray, int, list[float]] | tuple[np.ndarray, int]:
    """Test-time-compute: pick the best candidate per verifier.

    Returns (best_candidate, best_index[, all_scores]).
    """
    if not candidate_hvs:
        raise ValueError("no candidates to score")
    scores = [verifier.score(query_hv, c) for c in candidate_hvs]
    best_idx = int(np.argmax(scores))
    if return_scores:
        return candidate_hvs[best_idx], best_idx, scores
    return candidate_hvs[best_idx], best_idx


def self_consistency_vote(
    candidate_hvs: list[np.ndarray], threshold: float = 0.8
) -> tuple[np.ndarray, float]:
    """Pick the candidate that is most similar to the rest (consistency
    vote). Returns (winner, consistency_score).

    consistency_score = mean cosine of winner with all other candidates.
    """
    if not candidate_hvs:
        raise ValueError("no candidates to vote on")
    n = len(candidate_hvs)
    if n == 1:
        return candidate_hvs[0], 1.0
    stack = np.stack(candidate_hvs, axis=0)
    norm = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-9)
    sims = norm @ norm.T  # (n, n)
    np.fill_diagonal(sims, 0.0)
    mean_sims = sims.mean(axis=1)
    winner = int(np.argmax(mean_sims))
    return candidate_hvs[winner], float(mean_sims[winner])
