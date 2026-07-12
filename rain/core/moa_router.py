"""Mixture-of-Architectures (MoA) router.

This is the novel piece of RAIN-Net's composition. Standard
Mixture-of-Experts (Shazeer 2017, used in GPT-4 et al.) uses identical
transformer experts and differs only in which gets the gradient. RAIN-Net
uses architecturally heterogeneous experts -- each expert is a DIFFERENT
kind of model -- because different problems want different inductive
biases. The unification through the HV substrate (every expert reads/
writes HVs) is what makes this composition tractable.

The eight experts in v0.1:

    samba_lm          general language (HYMN-Samba base)
    pure_attn         long-range associative recall
    tsetlin           Boolean / rule-based reasoning
    diffusion         parallel generation, image-style decoding
    gnn               structured / graph reasoning
    sdm               sparse-distributed-memory episodic recall
    sym_regression    math / program induction
    jepa_wm           latent world-model prediction

Each expert has a learned "domain HV" that summarises the kind of query
it is good at. At routing time we compute cosine similarity between the
query HV and each domain HV, take top-k experts, and run them in
parallel. Their outputs are HVs which we bundle weighted by routing
score to produce the final answer HV.

Why this is cheaper than running a dense model:
    - Only top-k experts compute (k=2 or 3 typical).
    - Each expert is small (50-200M params each at v0.1 scale).
    - Routing cost is O(N_experts * D) ~ 80K cosines -- free.
    - Total active params per query: 2-3 small experts vs full LLM weights.

Why this is better-quality than a single dense model:
    - Each expert has the right inductive bias for its domain.
    - A query like "factor 1547" routes to sym_regression (which can
      actually do it) instead of asking a transformer to guess digits.
    - A query like "what was said 200 turns ago" routes to sdm
      (which remembers) instead of pushing past a transformer's window.
    - A query like "are these statements logically consistent" routes
      to tsetlin (which produces a proof) instead of relying on LM hand-
      waving.

Training: each expert trains independently on its specialty data.
Routing weights train via distillation against a teacher LLM's chosen
expert per query (collected via prompting the teacher with the expert
descriptions). Active learning updates the domain HVs over time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rain.core.hv_substrate import (
    DEFAULT_DIM,
    bundle_weighted,
    hash_to_hv,
    similarity_matrix,
)


# -------------------- expert interface --------------------


@dataclass
class ExpertSpec:
    """One expert in the MoA bank.

    Each expert wraps a callable that takes an HV (the query) and returns
    an HV (the answer). The callable can be any architecture -- this
    interface intentionally hides the implementation. v0.1 stubs use
    deterministic functions for testing; production swaps in real models.
    """

    name: str
    domain_hv: np.ndarray  # (D,) -- HV that summarises this expert's specialty
    forward: Callable[[np.ndarray], np.ndarray]  # query_hv -> answer_hv
    description: str = ""
    # Trainable counter -- how often this expert was routed and judged correct
    success_count: int = 0
    failure_count: int = 0


# -------------------- default expert constructors --------------------
# These are STUB implementations that exercise the routing logic + return
# plausible HVs. Production experts replace each forward fn with the
# actual model. The interface contract is (query_hv,) -> (answer_hv,).


def _make_stub_expert(name: str, dim: int, seed: int) -> ExpertSpec:
    """Stub expert: domain HV is hash-derived; forward applies a fixed
    deterministic transform so we can test routing + composition without
    training real sub-models. Replace with real expert in production.
    """
    domain_hv = hash_to_hv(f"expert::{name}::domain", dim=dim)
    transform_hv = hash_to_hv(f"expert::{name}::transform", dim=dim)

    def forward(query_hv: np.ndarray) -> np.ndarray:
        # Deterministic stub: bind query with the expert's transform HV.
        # Real experts would run their actual model here.
        from rain.core.hv_substrate import bind

        return bind(query_hv, transform_hv)

    return ExpertSpec(
        name=name,
        domain_hv=domain_hv,
        forward=forward,
        description=f"stub-{name} (replace with real model)",
    )


def default_expert_bank(dim: int = DEFAULT_DIM) -> list[ExpertSpec]:
    """The eight v0.1 experts as stubs. Replace forwards with real
    models as they become available."""
    names = [
        "samba_lm",
        "pure_attn",
        "tsetlin",
        "diffusion",
        "gnn",
        "sdm",
        "sym_regression",
        "jepa_wm",
    ]
    return [_make_stub_expert(n, dim, seed=i) for i, n in enumerate(names)]


# -------------------- router --------------------


@dataclass
class MoARouter:
    """Routes a query HV to top-k experts and aggregates their outputs.

    Parameters
    ----------
    experts        : list of ExpertSpec, one per expert
    top_k          : how many experts to run per query (1-3 typical)
    temperature    : softmax temperature on routing similarities; lower
                     = sharper routing (concentrate on top-1)
    """

    experts: list[ExpertSpec]
    top_k: int = 2
    temperature: float = 1.0
    _domain_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("experts list cannot be empty")
        if not (1 <= self.top_k <= len(self.experts)):
            raise ValueError(f"top_k must be in [1, {len(self.experts)}]; got {self.top_k}")
        self._rebuild_domain_matrix()

    def _rebuild_domain_matrix(self) -> None:
        self._domain_matrix = np.stack([e.domain_hv for e in self.experts], axis=0)

    def route(self, query_hv: np.ndarray) -> list[tuple[int, float]]:
        """Compute (expert_idx, weight) for the top_k experts.

        Returns list of length top_k, sorted by weight descending.
        Weights are softmax over the top-k similarities (not over all
        experts) so the chosen subset's weights sum to 1.0.
        """
        sims = similarity_matrix(query_hv[np.newaxis, :], self._domain_matrix)[0]
        # Top-k by similarity
        top_idx = np.argsort(-sims)[: self.top_k]
        top_sims = sims[top_idx]
        # Softmax over the top-k
        logits = top_sims / max(self.temperature, 1e-6)
        e = np.exp(logits - logits.max())
        weights = e / (e.sum() + 1e-9)
        return [(int(i), float(w)) for i, w in zip(top_idx, weights, strict=True)]

    def forward(self, query_hv: np.ndarray) -> tuple[np.ndarray, list[tuple[str, float]]]:
        """Run the routed experts on the query and bundle their outputs.

        Returns
        -------
        answer_hv     : (D,) bundled output of all routed experts
        provenance    : list of (expert_name, weight) used for this answer
        """
        routing = self.route(query_hv)
        expert_outputs: list[np.ndarray] = []
        weights: list[float] = []
        provenance: list[tuple[str, float]] = []
        for idx, w in routing:
            expert = self.experts[idx]
            ans = expert.forward(query_hv)
            expert_outputs.append(ans)
            weights.append(w)
            provenance.append((expert.name, w))
        answer = bundle_weighted(expert_outputs, weights)
        return answer, provenance

    def record_outcome(self, provenance: list[tuple[str, float]], success: bool) -> None:
        """Record judge outcome to update expert reliability stats.

        Used by active learning: experts that succeed often on a domain
        get their domain_hv slightly nudged toward the query type;
        experts that fail often get their domain_hv nudged away.
        """
        name_to_idx = {e.name: i for i, e in enumerate(self.experts)}
        for name, _w in provenance:
            if name not in name_to_idx:
                continue
            e = self.experts[name_to_idx[name]]
            if success:
                e.success_count += 1
            else:
                e.failure_count += 1

    def nudge_domains(
        self, query_hv: np.ndarray, provenance: list[tuple[str, float]], success: bool, eta: float = 0.01
    ) -> None:
        """Online learning step on the domain HVs.

        If success: nudge winning experts' domain HVs toward query HV.
        If failure: nudge them away. Bipolar re-quantise after.

        eta = 0.01 is the default learning rate; we want slow adaptation
        so routing remains stable.
        """
        from rain.core.hv_substrate import bipolarize

        name_to_idx = {e.name: i for i, e in enumerate(self.experts)}
        sign = 1.0 if success else -1.0
        for name, w in provenance:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            current = self.experts[idx].domain_hv.astype(np.float32)
            delta = sign * eta * w * query_hv.astype(np.float32)
            self.experts[idx].domain_hv = bipolarize(current + delta)
        self._rebuild_domain_matrix()

    def stats(self) -> list[dict[str, float | str]]:
        """Per-expert success/failure stats for monitoring."""
        out: list[dict[str, float | str]] = []
        for e in self.experts:
            total = e.success_count + e.failure_count
            rate = float(e.success_count) / total if total > 0 else 0.0
            out.append(
                {
                    "expert": e.name,
                    "success": e.success_count,
                    "failure": e.failure_count,
                    "rate": rate,
                }
            )
        return out
