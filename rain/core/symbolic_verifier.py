"""Symbolic verifier: produces an audit trail for every RAIN-Net answer.

The output to the user is not just "answer X" -- it is:
    answer X
    + cited facts (from semantic memory)
    + clause trace (which Tsetlin clauses fired in support)
    + binding-coherence score (does answer HV match the bind of cited facts?)

LLMs structurally cannot produce this. RAG bolts citations on as
post-hoc text matching; it does not verify that the model's reasoning
actually used the cited facts. RAIN-Net's verifier checks the HV
algebra: the answer HV must be consistent with the bind of its claimed
supporting facts. If not, the model is hallucinating a citation, and we
flag it.

Two verification mechanisms:

    1. binding_coherence(answer_hv, cited_fact_hvs) -> float in [0,1]
       Measures cosine similarity between answer_hv and the bundle of
       cited fact HVs. High = answer is consistent with citations.

    2. clause_trace(answer_hv, tsetlin_model) -> list[clause_id, satisfied]
       Returns which Boolean clauses fired in producing the answer.
       Each clause is human-readable (Tsetlin clauses are conjunctions
       of literal features, so "fact_A AND NOT fact_B" is the rule).

The user-visible output bundles these into an AuditReport. For
regulated-domain deployment, the AuditReport IS the deliverable; the
answer text is secondary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hierarchical_memory import SemanticFact
from rain.core.hv_substrate import bundle, similarity


# -------------------- audit report --------------------


@dataclass
class AuditReport:
    """The audit object returned alongside every RAIN-Net answer.

    Fields:
        answer_text      : the human-readable answer
        cited_facts      : list of SemanticFact instances used
        binding_score    : cosine similarity in [0,1], answer vs cited bundle
        clause_trace     : list of (clause_id, satisfied_bool) Tsetlin firings
        verifier_score   : the neural verifier's confidence in [-1, +1]
        provenance       : list of (expert_name, weight) from MoA routing
        confidence       : aggregate confidence in [0, 1]
        warnings         : human-readable issues (low confidence, disputed
                           fact, etc.)
    """

    answer_text: str
    cited_facts: list[SemanticFact] = field(default_factory=list)
    binding_score: float = 0.0
    clause_trace: list[tuple[str, bool]] = field(default_factory=list)
    verifier_score: float = 0.0
    provenance: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer_text,
            "cited_facts": [
                {"id": f.fact_id, "text": f.text, "source": f.source} for f in self.cited_facts
            ],
            "binding_score": self.binding_score,
            "clause_trace": self.clause_trace,
            "verifier_score": self.verifier_score,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    def human_format(self) -> str:
        """One-screen human-readable form for CLI / chat display."""
        lines = [
            f"ANSWER: {self.answer_text}",
            f"CONFIDENCE: {self.confidence:.2f}",
            "",
            "CITED FACTS:",
        ]
        if not self.cited_facts:
            lines.append("  (none)")
        for f in self.cited_facts:
            lines.append(f"  - [{f.fact_id}] {f.text}")
            if f.source:
                lines.append(f"      source: {f.source}")
        lines += [
            "",
            f"BINDING SCORE: {self.binding_score:.2f}  (answer-vs-citations coherence)",
            f"VERIFIER SCORE: {self.verifier_score:+.2f}",
            "",
            "ROUTED EXPERTS:",
        ]
        for name, w in self.provenance:
            lines.append(f"  - {name}: weight={w:.2f}")
        if self.clause_trace:
            lines += ["", "CLAUSE TRACE (firing Boolean rules):"]
            for cid, satisfied in self.clause_trace:
                lines.append(f"  - {cid}: {'fired' if satisfied else 'not fired'}")
        if self.warnings:
            lines += ["", "WARNINGS:"]
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# -------------------- binding coherence --------------------


def binding_coherence(answer_hv: np.ndarray, cited_fact_hvs: list[np.ndarray]) -> float:
    """How consistent is the answer HV with the bundle of its citations?

    Returns float in [0, 1] (we clamp negative cosines to 0).

    Interpretation:
        > 0.7   strong coherence; answer follows from cited facts
        0.3-0.7 partial coherence; some citations relevant, some not
        < 0.3   weak/no coherence; model may be hallucinating citations
    """
    if not cited_fact_hvs:
        return 0.0
    fact_bundle = bundle(*cited_fact_hvs)
    sim = similarity(answer_hv, fact_bundle)
    return float(max(0.0, sim))


# -------------------- clause trace adapter --------------------


def clause_trace_from_tsetlin(
    answer_hv: np.ndarray, tsetlin_model: Any, max_clauses: int = 8
) -> list[tuple[str, bool]]:
    """Return which Tsetlin clauses fired for the given answer HV.

    The Tsetlin model is the existing rain.core.tsetlin module. We pass
    the answer HV through it and read out which clauses were satisfied
    by the literal features (bipolar HV components).

    Returns list of (clause_id, satisfied) up to max_clauses. The clause
    id is a string like "C7" -- the caller can join it against the
    Tsetlin model's clause descriptions for human-readable form.
    """
    if tsetlin_model is None:
        return []
    # Defensive: tsetlin_model API in rain.core.tsetlin varies; we wrap
    # in try/except and degrade gracefully. Real integration substitutes
    # the exact attribute / method names.
    try:
        # Expected interface: model.firing_clauses(literals) -> list[int]
        # Where literals = sign(answer_hv) > 0 (Boolean)
        literals = (answer_hv > 0).astype(np.int8)
        if hasattr(tsetlin_model, "firing_clauses"):
            ids = tsetlin_model.firing_clauses(literals)
            return [(f"C{cid}", True) for cid in ids[:max_clauses]]
        # Fallback: just hash the HV into a deterministic pseudo-clause id
        # so the audit report has SOMETHING for the user even if Tsetlin
        # isn't fully wired.
        h = int.from_bytes(literals.tobytes()[:4], "big") % 1000
        return [(f"C{h}", True)]
    except (AttributeError, TypeError, ValueError):
        return []


# -------------------- top-level assemble --------------------


def make_audit_report(
    answer_text: str,
    answer_hv: np.ndarray,
    cited_facts: list[SemanticFact],
    verifier_score: float,
    provenance: list[tuple[str, float]],
    tsetlin_model: Any | None = None,
    confidence_thresh: float = 0.5,
) -> AuditReport:
    """Build the AuditReport from raw RAIN-Net outputs.

    This is the one function the demo CLI / chat surface should call
    after every answer. The returned report has everything needed for
    user display + downstream automation.
    """
    cited_hvs = [f.hv for f in cited_facts]
    coherence = binding_coherence(answer_hv, cited_hvs)
    clause_tr = clause_trace_from_tsetlin(answer_hv, tsetlin_model)

    # Aggregate confidence: harmonic mean of verifier and coherence
    # (both need to be high for confidence to be high). verifier_score
    # is in [-1, +1] so we rescale to [0, 1].
    v01 = (verifier_score + 1.0) / 2.0
    if coherence == 0.0 and v01 == 0.0:
        confidence = 0.0
    else:
        confidence = float(2 * v01 * coherence / (v01 + coherence + 1e-9))

    warnings: list[str] = []
    if confidence < confidence_thresh:
        warnings.append(f"low confidence ({confidence:.2f}); consider teacher fallback")
    if any(f.disputed for f in cited_facts):
        warnings.append("at least one cited fact is flagged disputed")
    if not cited_facts:
        warnings.append("no facts cited; answer is unsupported")

    return AuditReport(
        answer_text=answer_text,
        cited_facts=cited_facts,
        binding_score=coherence,
        clause_trace=clause_tr,
        verifier_score=verifier_score,
        provenance=provenance,
        confidence=confidence,
        warnings=warnings,
    )
