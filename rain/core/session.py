"""Multi-turn conversation session for RAIN-Net.

LLMs have a fixed context window. After it fills, old turns fall out and
the model literally cannot recall them. RAIN-Net's hierarchical memory
stores every turn in episodic memory addressed by HV cosine, so older
turns remain retrievable by semantic similarity for arbitrary session
length.

A Session wraps a RainNet, tracks turn count, records each (query,
answer) pair in episodic memory, and exposes a `recall(query)` method
that pulls relevant past turns from the episodic bank. The session is
also responsible for stitching past-turn context into new-turn answers
when relevant.

Use:
    session = Session(net=RainNet())
    r1 = session.turn("What is photosynthesis?")
    r2 = session.turn("How does it relate to respiration?")
    # Session knows about turn 1 via episodic recall.
    r3 = session.turn("What did you say earlier about plants?")
    # Recall surfaces the photosynthesis turn even after many turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rain.core.hv_substrate import bundle, similarity_matrix
from rain.core.rain_net import RainNet
from rain.core.symbolic_verifier import AuditReport


@dataclass
class Turn:
    """One turn of conversation."""

    turn_id: int
    query: str
    answer_text: str
    confidence: float
    cited_fact_ids: list[str] = field(default_factory=list)
    routed_experts: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Session:
    """Multi-turn conversation wrapper around a RainNet.

    Properties:
        net           : the underlying RainNet (provides KB + routing + verifier)
        turns         : list of Turn objects, in order
        recall_top_k  : how many past turns to recall when stitching context
    """

    net: RainNet
    recall_top_k: int = 3
    turns: list[Turn] = field(default_factory=list)
    _next_id: int = 0

    def turn(self, query: str, modality: str = "text") -> AuditReport:
        """Run one turn. The query is answered as-is; episodic memory
        is consulted SEPARATELY (not via query-string augmentation,
        which would contaminate skill / routing decisions).

        Returns the AuditReport for this turn, with a leading warning
        listing which past turn ids were recalled (for transparency).
        """
        # Compute the recall list for transparency, but do NOT mutate
        # the query. The underlying RainNet has access to episodic
        # memory directly via memory.episodic; it will surface relevant
        # past turns through normal HV-similarity lookups.
        recalled_turns = self.recall(query, top_k=self.recall_top_k)

        # Ask the net with the original query (untouched).
        report = self.net.answer(query, modality=modality)

        # Record this turn.
        t = Turn(
            turn_id=self._next_id,
            query=query,
            answer_text=report.answer_text,
            confidence=report.confidence,
            cited_fact_ids=[f.fact_id for f in report.cited_facts],
            routed_experts=report.provenance,
        )
        self.turns.append(t)
        self._next_id += 1

        # Attach a leading recall note to the report's warnings for
        # visibility. This does not alter the answer; it only annotates.
        report.warnings = list(report.warnings)
        if recalled_turns:
            recalled_ids = [pt.turn_id for _s, pt in recalled_turns]
            report.warnings.insert(
                0, f"recalled prior turns {recalled_ids} from episodic memory"
            )
        return report

    def recall(self, query: str, top_k: int = 3) -> list[tuple[float, Turn]]:
        """Pull the most semantically-related past turns for a query.

        Returns list of (similarity_score, Turn) sorted descending.
        """
        if not self.turns:
            return []
        # Encode the query and each past turn's query.
        q_hv = self.net.encoder_bank.encode("text", query)
        bank_hvs = [
            self.net.encoder_bank.encode("text", t.query) for t in self.turns
        ]
        # Stack and compute cosines.
        import numpy as np

        bank = np.stack(bank_hvs, axis=0)
        sims = similarity_matrix(q_hv[np.newaxis, :], bank)[0]
        # Top-k by score.
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.turns[i]) for i in idx]

    def history_summary(self) -> str:
        """Human-readable summary of the session for display."""
        if not self.turns:
            return "(empty session)"
        lines = [f"Session has {len(self.turns)} turn(s):"]
        for t in self.turns:
            q = t.query[:60] + ("..." if len(t.query) > 60 else "")
            a = t.answer_text[:60] + ("..." if len(t.answer_text) > 60 else "")
            lines.append(f"  [t{t.turn_id}] Q: {q}")
            lines.append(f"          A: {a}  (conf={t.confidence:.2f})")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        return {
            "n_turns": len(self.turns),
            "kb_size": len(self.net.memory.semantic),
            "mean_confidence": (
                sum(t.confidence for t in self.turns) / len(self.turns)
                if self.turns
                else 0.0
            ),
            "episodic_records": len(self.net.memory.episodic),
        }
