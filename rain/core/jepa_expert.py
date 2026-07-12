"""JEPA-style world-model expert (replaces the last stub expert).

Inspired by Yann LeCun's Joint-Embedding Predictive Architecture (JEPA,
2022). Instead of predicting raw next tokens, JEPA predicts the latent
embedding of the next state. RAIN-Net's substrate-aware JEPA does the
same: given a query HV (current state), predict the answer-shaped HV
(next state) directly in the 10K-D HV space.

Two roles:
    1. As an MoA expert: route queries here when the model needs to
       "predict what the answer should look like" rather than retrieve
       it from KB. Useful for queries with no direct KB match.
    2. As a verifier signal: the predicted answer HV from JEPA can be
       compared to the retrieved answer HV; large divergence -> low
       confidence -> trigger reflexion or teacher escalation.

Training: online from (query, answer) pairs in episodic memory.
The predictor is a single per-position permutation map P_k such that:
    predicted_answer_hv ~ bundle_over_k(permute(query_hv, k) * W_k)
where W_k is a learned weight HV per position. This is a tiny model
(K * D bipolar bits) but captures the "what kind of answer" structure
of the query without LM-style generation.

Falls back to a deterministic transform if no training data yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hv_substrate import (
    DEFAULT_DIM,
    bind,
    bundle,
    hash_to_hv,
    permute,
    similarity,
)
from rain.core.moa_router import ExpertSpec


@dataclass
class JEPAState:
    """Internal state for the JEPA predictor."""

    dim: int = DEFAULT_DIM
    n_permutations: int = 8
    learning_rate: float = 0.1
    weights: np.ndarray = field(init=False)  # (K, D) float
    n_updates: int = 0
    last_predicted: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Init weights to small random; they'll be updated by .train()
        rng = np.random.default_rng(7)
        self.weights = (rng.integers(0, 2, size=(self.n_permutations, self.dim)) * 2 - 1).astype(np.float32)

    def predict(self, query_hv: np.ndarray) -> np.ndarray:
        """Predict the answer HV for a query.

        prediction = bundle_k( bind(weights[k], permute(query, k)) )
        """
        parts = []
        for k in range(self.n_permutations):
            parts.append(bind(self.weights[k], permute(query_hv, k)))
        pred = bundle(*parts)
        self.last_predicted = pred
        return pred

    def train_step(self, query_hv: np.ndarray, target_hv: np.ndarray) -> None:
        """Online perceptron-style update.

        Move weights[k] toward (target unbind permute(query, k)) so the
        predictor's bundle moves toward target.

        target = sum_k bind(weights_new[k], permute(query, k))
        => For each k: weights_new[k] = unbind(target, permute(query, k))
        Average over k via additive-update + bipolar normalisation.
        """
        for k in range(self.n_permutations):
            ideal_k = bind(target_hv, permute(query_hv, k))  # if this were a single-permutation model
            # Perceptron step toward ideal_k.
            self.weights[k] = self.weights[k] + self.learning_rate * ideal_k
        self.n_updates += 1


def make_jepa_expert(
    dim: int = DEFAULT_DIM,
    n_permutations: int = 8,
    learning_rate: float = 0.1,
) -> ExpertSpec:
    """Real JEPA world-model expert. Replaces the stub.

    Forward path: predict answer HV from query HV. The expert can also
    be trained online (each successful (query, answer) pair calls
    state.train_step). Wiring of online training is left to RainNet's
    feedback() / active-learning hooks in v0.3.
    """
    from rain.core.real_experts import _seeded_domain_hv

    domain_hv = _seeded_domain_hv("jepa_wm", dim=dim)
    state = JEPAState(dim=dim, n_permutations=n_permutations, learning_rate=learning_rate)

    def forward(query_hv: np.ndarray) -> np.ndarray:
        pred = state.predict(query_hv)
        # Bind with query so the expert output carries query context (per
        # convention with the other real experts).
        return bind(query_hv, pred)

    spec = ExpertSpec(
        name="jepa_wm",
        domain_hv=domain_hv,
        forward=forward,
        description=f"real JEPA world-model expert (K={n_permutations} permutations)",
    )
    # Stash state on the spec so RainNet can train it.
    spec._jepa_state = state  # type: ignore[attr-defined]
    return spec


def train_jepa_from_episodic(spec: ExpertSpec, episodic_memory) -> int:
    """Bulk-train the JEPA expert from the episodic memory's (query, answer)
    history. Returns the number of training steps performed.

    Call this periodically to keep the world-model in sync with the
    actual (query, answer) distribution the system sees.
    """
    state = getattr(spec, "_jepa_state", None)
    if state is None:
        return 0
    n = 0
    for entry in episodic_memory._entries:
        state.train_step(entry.query_hv, entry.answer_hv)
        n += 1
    return n
