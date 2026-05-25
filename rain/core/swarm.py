# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Multi-agent VSA-routed swarm of specialist RainNets.

Each swarm member is a full RainNet instance specialized to one domain
(by virtue of which facts were ingested into its semantic memory). The
manager holds a domain identity HV per member; routing is a single
HV cosine against the bank of domain identities.

Why this is novel against LangChain-style agent swarms:
    - LangChain routes via LLM-as-router (slow, opaque, expensive).
      RAIN-Net swarm routes via HV cosine (fast, transparent, free).
    - LangChain aggregates via LLM-as-merger (more LLM calls).
      RAIN-Net swarm aggregates via verifier vote OR HV bundle in
      the substrate (zero LLM calls).
    - LangChain agents are independent processes; RAIN-Net members
      can run in the same process and share encoder + verifier weights.
    - Horizontal scale: adding a new specialist is one RainNet + one
      domain HV. No router retraining.

This is the architectural piece that makes the Series A "scale via
composition not training" story concrete. To add a new vertical, you
add a new RainNet member with its own KB. The router learns nothing;
the HV similarity does the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hv_substrate import (
    DEFAULT_DIM,
    bundle_weighted,
    similarity_matrix,
)
from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.symbolic_verifier import AuditReport, make_audit_report
from rain.core.verifier_head import HVVerifierHead


@dataclass
class SwarmMember:
    """One specialist RainNet plus its domain identity HV."""

    name: str
    net: RainNet
    domain_hv: np.ndarray
    domain_text: str = ""
    # Usage stats for monitoring
    invocations: int = 0
    success_count: int = 0


@dataclass
class SwarmRouting:
    """One routing decision: top-k members selected for a query."""

    members: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class SwarmAnswer:
    """Aggregated answer from multiple swarm members."""

    text: str
    contributing_members: list[tuple[str, AuditReport]]
    aggregated_hv: np.ndarray
    confidence: float


class RainNetSwarm:
    """A bank of specialist RainNet instances with HV-routed query
    dispatch + verifier-voted answer aggregation.

    Use:
        swarm = RainNetSwarm(dim=10000)
        swarm.add_member("aviation", domain_text="aircraft FAA AD")
        swarm.add_member("legal", domain_text="contract law clause")
        # Each member is a full RainNet; ingest into them directly:
        swarm["aviation"].net.ingest_fact("AD 2024-01-05 applies to C172...")
        # Route a query to top-2 swarm members + aggregate:
        result = swarm.answer("Which AD applies to my Cessna?", top_k=2)
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self.members: list[SwarmMember] = []
        self._domain_bank: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        # Shared resources to avoid duplicate model loads.
        self._shared_encoder = EncoderBank(dim=dim)
        self._shared_verifier = HVVerifierHead(dim=dim)

    # -------- membership --------

    def add_member(
        self,
        name: str,
        domain_text: str,
        net: RainNet | None = None,
        config: RainNetConfig | None = None,
    ) -> SwarmMember:
        """Register a new specialist. If net is None, build a fresh
        RainNet using the given config (or default config with this swarm's dim).
        """
        if any(m.name == name for m in self.members):
            raise ValueError(f"swarm member {name!r} already exists")
        if net is None:
            cfg = config or RainNetConfig(dim=self.dim, n_candidates=2, semantic_top_k=4)
            net = RainNet(config=cfg)
        # Domain HV = encoding of the descriptive text.
        domain_hv = self._shared_encoder.encode("text", domain_text)
        member = SwarmMember(name=name, net=net, domain_hv=domain_hv, domain_text=domain_text)
        self.members.append(member)
        self._rebuild_domain_bank()
        return member

    def remove_member(self, name: str) -> bool:
        """Remove a swarm member by name. Returns True if removed."""
        before = len(self.members)
        self.members = [m for m in self.members if m.name != name]
        removed = len(self.members) < before
        if removed:
            self._rebuild_domain_bank()
        return removed

    def __getitem__(self, name: str) -> SwarmMember:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(f"no swarm member named {name!r}")

    def __len__(self) -> int:
        return len(self.members)

    # -------- routing --------

    def route(self, query_hv: np.ndarray, top_k: int = 2) -> SwarmRouting:
        """Pick top-k swarm members by HV cosine against their domain HV."""
        if not self.members:
            return SwarmRouting(members=[])
        sims = similarity_matrix(query_hv[np.newaxis, :], self._domain_bank)[0]
        k = min(top_k, len(self.members))
        top_idx = np.argsort(-sims)[:k]
        # Softmax over the top-k.
        top_sims = sims[top_idx]
        e = np.exp(top_sims - top_sims.max())
        weights = e / (e.sum() + 1e-9)
        return SwarmRouting(
            members=[
                (self.members[i].name, float(w))
                for i, w in zip(top_idx, weights, strict=True)
            ]
        )

    # -------- answer --------

    def answer(
        self,
        query: str,
        top_k: int = 2,
        modality: str = "text",
    ) -> SwarmAnswer:
        """Route + dispatch + aggregate. Returns a SwarmAnswer with the
        consensus text + per-member reports.

        Aggregation strategy:
            - Pull each routed member's AuditReport for the query.
            - Bundle their answer-shaped HVs weighted by routing score.
            - Pick the highest-confidence member's text as the consensus
              (verifier vote could replace this in v0.3).
        """
        if not self.members:
            from rain.core.symbolic_verifier import AuditReport as _AR
            return SwarmAnswer(
                text="(empty swarm)",
                contributing_members=[],
                aggregated_hv=np.zeros(self.dim, dtype=np.float32),
                confidence=0.0,
            )

        query_hv = self._shared_encoder.encode(modality, query)
        routing = self.route(query_hv, top_k=top_k)

        # Dispatch in sequence (parallelism would need threads; v0.2 keeps
        # it serial for determinism + simplicity).
        contributions: list[tuple[str, AuditReport, float]] = []
        for member_name, weight in routing.members:
            member = self[member_name]
            member.invocations += 1
            try:
                report = member.net.answer(query, modality=modality)
                contributions.append((member_name, report, weight))
            except Exception:  # noqa: BLE001
                continue

        if not contributions:
            return SwarmAnswer(
                text="(all routed members failed)",
                contributing_members=[],
                aggregated_hv=np.zeros(self.dim, dtype=np.float32),
                confidence=0.0,
            )

        # Aggregate answer HVs.
        hvs: list[np.ndarray] = []
        weights: list[float] = []
        for _name, report, w in contributions:
            # Use the bundle of cited fact HVs as the member's answer HV
            # (proxy; production would store the actual answer HV per report).
            if report.cited_facts:
                from rain.core.hv_substrate import bundle

                ans_hv = bundle(*[f.hv for f in report.cited_facts])
            else:
                ans_hv = query_hv
            hvs.append(ans_hv)
            weights.append(w)
        aggregated_hv = bundle_weighted(hvs, weights)

        # Pick the consensus text: highest member confidence.
        best = max(contributions, key=lambda c: c[1].confidence)
        consensus_text = best[1].answer_text
        consensus_conf = best[1].confidence
        # Mark the winning member as a success contributor.
        winning = next((m for m in self.members if m.name == best[0]), None)
        if winning is not None:
            winning.success_count += 1

        return SwarmAnswer(
            text=consensus_text,
            contributing_members=[(n, r) for n, r, _w in contributions],
            aggregated_hv=aggregated_hv,
            confidence=consensus_conf,
        )

    # -------- monitoring --------

    def stats(self) -> dict[str, Any]:
        return {
            "n_members": len(self.members),
            "members": [
                {
                    "name": m.name,
                    "domain_text": m.domain_text[:60],
                    "kb_size": len(m.net.memory.semantic),
                    "invocations": m.invocations,
                    "successes": m.success_count,
                    "success_rate": (
                        m.success_count / m.invocations if m.invocations > 0 else 0.0
                    ),
                }
                for m in self.members
            ],
        }

    # -------- internals --------

    def _rebuild_domain_bank(self) -> None:
        if not self.members:
            self._domain_bank = np.zeros((0, self.dim), dtype=np.float32)
            return
        self._domain_bank = np.stack([m.domain_hv for m in self.members], axis=0)
