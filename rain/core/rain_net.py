# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN-Net: the full composition.

This is the public API. Everything else (HV substrate, encoder bank, MoA
router, hierarchical memory, verifier head, symbolic verifier) is wired
together here behind two methods:

    answer(query, modality='text', candidates=8) -> AuditReport

    ingest_fact(fact_id, text, source='') -> None

The first is the user-facing query loop. The second is how knowledge
enters the system at runtime (no retraining).

This is intentionally simple. The complexity lives in the modules. By
the time you get here, every piece has a well-defined contract and a
tested implementation, so the composition is just plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hierarchical_memory import HierarchicalMemory, SemanticFact
from rain.core.hv_substrate import DEFAULT_DIM, bundle_weighted, hash_to_hv
from rain.core.moa_router import ExpertSpec, MoARouter, default_expert_bank
from rain.core.procedural_adapter import SkillRegistry
from rain.core.real_experts import install_text_sidechannel, make_real_expert_bank
from rain.core.symbolic_verifier import AuditReport, make_audit_report
from rain.core.verifier_head import HVVerifierHead, sample_and_select


# -------------------- config --------------------


@dataclass
class RainNetConfig:
    """Top-level config for a RAIN-Net instance.

    Most knobs default to v0.1 sane settings; only `dim` is commonly
    tuned (lower for speed, higher for capacity).
    """

    dim: int = DEFAULT_DIM
    n_candidates: int = 8  # TTC sample count
    router_top_k: int = 2
    router_temperature: float = 1.0
    semantic_top_k: int = 5
    confidence_threshold: float = 0.5  # below this, flag for teacher fallback
    use_real_experts: bool = True  # if False, use deterministic stubs
    hymn_checkpoint_path: str | None = None  # path to trained HYMN-Plus .npz
    verifier_checkpoint_path: str | None = None  # path to trained verifier .npz
    skills_dir: str | None = None  # path to procedural-skill dir (loads all sub-dirs)
    skill_match_threshold: float = 0.25  # min cosine to trigger a skill


# -------------------- main class --------------------


@dataclass
class RainNet:
    """The composition. Holds an encoder bank, an MoA router, a hierarchical
    memory, a verifier head, and produces audited answers.

    Use:
        net = RainNet()
        net.ingest_fact('f1', 'Apollo 11 landed on the Moon in 1969.')
        report = net.answer('When did Apollo 11 land on the Moon?')
        print(report.human_format())
    """

    config: RainNetConfig = field(default_factory=RainNetConfig)
    encoder_bank: EncoderBank = field(init=False)
    router: MoARouter = field(init=False)
    memory: HierarchicalMemory = field(init=False)
    verifier: HVVerifierHead = field(init=False)
    _next_fact_id: int = 0
    _tsetlin: Any = None  # set externally if integration desired

    def __post_init__(self) -> None:
        d = self.config.dim
        self.encoder_bank = EncoderBank(dim=d)
        if self.config.use_real_experts:
            experts = make_real_expert_bank(
                dim=d, hymn_checkpoint=self.config.hymn_checkpoint_path
            )
        else:
            experts = default_expert_bank(dim=d)
        self.router = MoARouter(
            experts=experts,
            top_k=self.config.router_top_k,
            temperature=self.config.router_temperature,
        )
        self.memory = HierarchicalMemory(dim=d)
        # Load trained verifier from disk if checkpoint path provided + exists.
        v_path = self.config.verifier_checkpoint_path
        if v_path:
            from pathlib import Path as _Path

            if _Path(v_path).exists():
                try:
                    self.verifier = HVVerifierHead.load(v_path)
                except (OSError, ValueError, KeyError):
                    self.verifier = HVVerifierHead(dim=d)
            else:
                self.verifier = HVVerifierHead(dim=d)
        else:
            self.verifier = HVVerifierHead(dim=d)
        # Auto-attach the Tsetlin machine to symbolic_verifier if any
        # real Tsetlin expert is present in the bank.
        for e in experts:
            if hasattr(e, "_tsetlin_machine"):
                self._tsetlin = e._tsetlin_machine
                break
        # Load procedural skills from disk if a skills_dir is configured.
        self.skill_registry = SkillRegistry()
        if self.config.skills_dir:
            self.skill_registry.load_dir(self.config.skills_dir)

    # ---------- public API ----------

    def answer(self, query: str, modality: str = "text", n_candidates: int | None = None) -> AuditReport:
        """The main query method. Returns an AuditReport.

        n_candidates overrides config.n_candidates for this call.
        """
        n = n_candidates if n_candidates is not None else self.config.n_candidates
        query_hv = self.encoder_bank.encode(modality, query)

        # Install text side-channel so experts that need raw text (e.g.
        # sym_regression for arithmetic, samba_lm for LM generation) can
        # use it. Pure-HV experts ignore this.
        if modality == "text":
            install_text_sidechannel(self.router.experts, query)

        # 0. Skill-router check: if a procedural skill matches strongly,
        # invoke it directly. Skills give deterministic-correct answers
        # for tasks they specialise in (e.g. arithmetic) so they
        # short-circuit the noisy LM + KB lookup.
        skill_match = self.skill_registry.match(
            query_hv,
            threshold=self.config.skill_match_threshold,
            query_text=query if modality == "text" else "",
        )
        skill_answer = None
        if skill_match:
            score, skill = skill_match[0]
            try:
                skill_answer = skill.invoke(query, kb=self.memory.semantic)
                self.memory.procedural.record_invocation(skill.meta.name)
            except Exception:  # noqa: BLE001
                skill_answer = None

        # 1. Retrieve from semantic memory.
        retrieved = self.memory.semantic.search(
            query_hv, top_k=self.config.semantic_top_k
        )
        cited_facts = [f for _s, f in retrieved]

        # 2. Run N candidate routings through MoA router.
        candidates: list[np.ndarray] = []
        provenances: list[list[tuple[str, float]]] = []
        for _ in range(n):
            ans_hv, prov = self.router.forward(query_hv)
            candidates.append(ans_hv)
            provenances.append(prov)

        # 3. Pick best via verifier head.
        best_hv, best_idx, all_scores = sample_and_select(
            candidates, query_hv, self.verifier, return_scores=True
        )
        verifier_score = all_scores[best_idx]
        chosen_provenance = provenances[best_idx]

        # 4. Synthesise human-readable answer text.
        # If a skill produced a valid answer, prefer it (deterministic).
        # We detect a "graceful failure" by either the universal prefix
        # convention "<skill_name>: could not" OR an empty string.
        skill_failed = (
            skill_answer is None
            or not skill_answer.strip()
            or "could not" in skill_answer.lower()
            or "no handler" in skill_answer.lower()
            or "error" in skill_answer.lower()[:80]
        )
        if not skill_failed and skill_answer is not None:
            answer_text = skill_answer
            score, skill = skill_match[0]
            chosen_provenance = chosen_provenance + [
                (f"skill::{skill.meta.name}", float(score))
            ]
        else:
            answer_text = self._synthesise_answer_text(query, cited_facts)

        # 5. Build audit report.
        report = make_audit_report(
            answer_text=answer_text,
            answer_hv=best_hv,
            cited_facts=cited_facts,
            verifier_score=verifier_score,
            provenance=chosen_provenance,
            tsetlin_model=self._tsetlin,
            confidence_thresh=self.config.confidence_threshold,
        )

        # 6. Record episode.
        self.memory.episodic.record(query_hv, best_hv, metadata={"query": query})

        return report

    def ingest_fact(
        self,
        text: str,
        source: str = "",
        fact_id: str | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Add a new fact to semantic memory. Returns the fact_id.

        This is the continual-learning entry point. No retraining needed.
        Encode the text into an HV, add to the semantic bank, done.
        """
        if fact_id is None:
            fact_id = f"f{self._next_fact_id}"
            self._next_fact_id += 1
        hv = self.encoder_bank.encode("text", text)
        self.memory.semantic.add(
            fact_id=fact_id, hv=hv, text=text, source=source, confidence=confidence
        )
        return fact_id

    def ingest_skill(
        self, skill_id: str, description: str, adapter_path: str, sample_queries: list[str]
    ) -> None:
        """Add a compiled skill keyed by the HV bundle of sample queries.

        sample_queries: list of text strings exemplifying queries this
        skill handles. We encode each, bundle, and use as the skill's
        domain HV.
        """
        if not sample_queries:
            raise ValueError("sample_queries required to define skill domain")
        hvs = [self.encoder_bank.encode("text", q) for q in sample_queries]
        weights = [1.0 / len(hvs)] * len(hvs)
        domain_hv = bundle_weighted(hvs, weights)
        self.memory.procedural.add(
            skill_id=skill_id,
            domain_hv=domain_hv,
            adapter_path=adapter_path,
            description=description,
        )

    def feedback(
        self, query: str, answer_hv: np.ndarray, correct: bool, modality: str = "text"
    ) -> None:
        """Online feedback from judge or user. Updates verifier head and
        router domain HVs. Never touches base experts."""
        query_hv = self.encoder_bank.encode(modality, query)
        label = 1 if correct else -1
        self.verifier.update(query_hv, answer_hv, label)
        # Router stat update (the most recent routing's provenance is
        # not tracked here for simplicity; caller could pass it in).
        # Nudge happens in active_learning.py with full provenance.

    def stats(self) -> dict[str, Any]:
        """Diagnostic snapshot for monitoring / dashboards."""
        return {
            "memory": self.memory.stats(),
            "verifier_updates": self.verifier.n_updates,
            "router": self.router.stats(),
            "encoder_modalities": self.encoder_bank.list_modalities(),
        }

    # ---------- internals ----------

    def _synthesise_answer_text(self, query: str, cited_facts: list[SemanticFact]) -> str:
        """Produce human-readable answer text.

        Strategy:
            1. If the HYMN samba_lm expert has produced fluent generated
               text this turn, use it (real LM output, conditioned on
               the query).
            2. Else, fall back to citing facts directly.

        The samba_lm expert stashes its last generation in
        spec._state['last_generated'] during its forward() call.
        """
        # Prefer LM-generated text if available.
        for e in self.router.experts:
            if e.name == "samba_lm":
                s = getattr(e, "_state", None)
                if isinstance(s, dict):
                    gen = s.get("last_generated", "").strip()
                    if gen:
                        # Combine LM continuation with cited facts for
                        # grounded output.
                        if cited_facts:
                            top = cited_facts[0].text
                            return f"{gen} (grounded in: {top})"
                        return gen
                break
        # Fallback: citation-based answer.
        if not cited_facts:
            return f"I don't have grounded information to answer: {query}"
        if len(cited_facts) == 1:
            return cited_facts[0].text
        parts = "; ".join(f.text for f in cited_facts[:3])
        return f"Based on retrieved facts: {parts}"
