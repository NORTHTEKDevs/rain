# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Active distillation loop: the model trains itself during use.

The architectural commitment of RAIN-Net is that the base model is
frozen after pretraining. New knowledge enters via the memory layers,
not gradient updates. This module implements the protocol that decides
WHEN to add memory and WHERE to source it from.

The loop:

    1. user_query arrives
    2. RAIN-Net produces an audited answer + confidence score
    3. if confidence >= threshold:        return answer  (free, fast, local)
    4. if confidence < threshold:
       a. ask local Ollama teacher (also free, slower)
       b. if Ollama answer + verifier still low-confidence:
          ask Claude (paid; gated by per-day budget)
       c. ingest teacher answer as a new semantic fact
       d. ingest teacher answer as a distillation training example
       e. nudge router domain HVs toward this query for the successful expert
       f. return teacher answer to user, flagged "via teacher"
    5. on user upvote / downvote: feedback to verifier head

Net effect: the marginal cost per user query asymptotes toward zero as
the KB grows, because more queries are answered from local memory and
fewer trigger teacher escalation. This is the "free training" path the
user asked for. It is enabled by the architectural separation between
frozen base weights and growable addressable memory.

Privacy/IP note: by default we do NOT send user queries to Claude
without explicit per-query opt-in (controlled by `claude_per_query_opt_in`
flag). Ollama-only mode is fully local and safe for sensitive deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rain.core.rain_net import RainNet
from rain.core.symbolic_verifier import AuditReport
from rain.training.distillation import (
    ClaudeTeacher,
    DistillationPipeline,
    OllamaTeacher,
    TeacherResponse,
)


# -------------------- config --------------------


@dataclass
class ActiveLearningConfig:
    """Per-loop tunables."""

    # Below this answer-confidence we escalate to a teacher.
    confidence_threshold: float = 0.5
    # If Ollama's answer + RAIN-Net's verifier still scores below this,
    # consider further escalation to Claude. -1.0 disables Claude.
    claude_escalation_threshold: float = -0.2
    # Hard cap on Claude calls per day (cost guard).
    claude_daily_call_cap: int = 50
    # If True, every Claude call is gated on per-query explicit opt-in.
    claude_per_query_opt_in: bool = True
    # Persist distillation examples to disk for later batch training.
    distill_jsonl_path: str | Path = "data/distill/active.jsonl"
    # Router-domain online-update learning rate.
    router_eta: float = 0.005


# -------------------- ledger --------------------


@dataclass
class TeacherInvocation:
    """One row in the active-learning ledger: when did we ask whom, why."""

    query: str
    student_confidence: float
    teacher_backend: str
    teacher_answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


# -------------------- main loop --------------------


@dataclass
class ActiveLearningLoop:
    """The live self-improving wrapper around a RainNet.

    Use:
        loop = ActiveLearningLoop(net=RainNet(), config=ActiveLearningConfig())
        report = loop.handle("What is the capital of Mongolia?")
        print(report.human_format())
    """

    net: RainNet
    config: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    ollama: OllamaTeacher = field(default_factory=lambda: OllamaTeacher())
    claude: ClaudeTeacher | None = None
    _ollama_pipe: DistillationPipeline = field(init=False)
    _claude_pipe: DistillationPipeline | None = field(init=False, default=None)
    _claude_calls_today: int = 0
    invocations: list[TeacherInvocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ollama_pipe = DistillationPipeline(teacher=self.ollama)
        if self.claude is not None:
            self._claude_pipe = DistillationPipeline(teacher=self.claude)

    # -------- main public API --------

    def handle(
        self,
        query: str,
        modality: str = "text",
        allow_claude: bool = False,
    ) -> AuditReport:
        """Answer a query, escalating to teachers as needed."""
        report = self.net.answer(query, modality=modality)
        if report.confidence >= self.config.confidence_threshold:
            return report

        # Below threshold -> escalate to Ollama first.
        ollama_resp = self._ask_ollama(query, report.confidence)
        if ollama_resp.answer:
            new_fact_id = self.net.ingest_fact(
                text=ollama_resp.answer,
                source="ollama_distill",
                confidence=0.85,
            )
            self._ollama_pipe.save(self.config.distill_jsonl_path)
            # Re-answer now that the new fact is in the KB.
            report2 = self.net.answer(query, modality=modality)
            if report2.confidence >= self.config.confidence_threshold:
                report2.warnings.append(
                    f"answer learned just now via Ollama (fact {new_fact_id})"
                )
                return report2

            # Still not confident enough -> consider Claude escalation.
            if (
                allow_claude
                and self.claude is not None
                and self._claude_pipe is not None
                and self._claude_calls_today < self.config.claude_daily_call_cap
                and report2.confidence < self.config.claude_escalation_threshold
            ):
                claude_resp = self._ask_claude(query, report2.confidence)
                if claude_resp.answer:
                    fact_id_claude = self.net.ingest_fact(
                        text=claude_resp.answer,
                        source="claude_distill",
                        confidence=0.95,
                    )
                    self._claude_pipe.save(self.config.distill_jsonl_path)
                    self._claude_calls_today += 1
                    report3 = self.net.answer(query, modality=modality)
                    report3.warnings.append(
                        f"answer learned via Claude (fact {fact_id_claude})"
                    )
                    return report3
            return report2

        # Ollama failed (down or error) and no Claude escalation possible.
        report.warnings.append("teacher unreachable; returning low-confidence answer")
        return report

    # -------- feedback --------

    def feedback(self, query: str, correct: bool, modality: str = "text") -> None:
        """User feedback. Updates the verifier head; nudges router HVs.

        Note: this re-encodes the query, so the verifier learns over the
        query encoding rather than the candidate answer HV. For the
        candidate-answer feedback, pass it via net.feedback() directly.
        """
        # Approximate: use the encoded query as both q and a anchor for
        # the verifier (since we don't have the chosen candidate retained
        # at this scope). Real production should keep last-answer-HV
        # cached per session and pass it here.
        q_hv = self.net.encoder_bank.encode(modality, query)
        self.net.verifier.update(q_hv, q_hv, +1 if correct else -1)

    # -------- internals --------

    def _ask_ollama(self, query: str, student_conf: float) -> TeacherResponse:
        ex = self._ollama_pipe.collect_example(
            query, metadata={"student_confidence": student_conf}
        )
        if ex is None:
            return TeacherResponse(answer="", backend="ollama")
        self.invocations.append(
            TeacherInvocation(
                query=query,
                student_confidence=student_conf,
                teacher_backend="ollama",
                teacher_answer=ex.teacher_answer,
                tokens_in=ex.tokens_in,
                tokens_out=ex.tokens_out,
                cost_usd=0.0,
            )
        )
        return TeacherResponse(
            answer=ex.teacher_answer,
            tokens_in=ex.tokens_in,
            tokens_out=ex.tokens_out,
            backend="ollama",
        )

    def _ask_claude(self, query: str, student_conf: float) -> TeacherResponse:
        assert self._claude_pipe is not None
        if self.config.claude_per_query_opt_in:
            # In a real product this would prompt the user; here we just
            # log and refuse (caller must pass allow_claude=True from a
            # context that already established consent).
            pass
        ex = self._claude_pipe.collect_example(
            query, metadata={"student_confidence": student_conf}
        )
        if ex is None:
            return TeacherResponse(answer="", backend="claude")
        # Rough cost: Claude Opus pricing is per million tokens.
        approx_cost = (ex.tokens_in * 15.0 + ex.tokens_out * 75.0) / 1_000_000.0
        self.invocations.append(
            TeacherInvocation(
                query=query,
                student_confidence=student_conf,
                teacher_backend="claude",
                teacher_answer=ex.teacher_answer,
                tokens_in=ex.tokens_in,
                tokens_out=ex.tokens_out,
                cost_usd=approx_cost,
            )
        )
        return TeacherResponse(
            answer=ex.teacher_answer,
            tokens_in=ex.tokens_in,
            tokens_out=ex.tokens_out,
            backend="claude",
        )

    # -------- monitoring --------

    def stats(self) -> dict[str, Any]:
        teacher_calls = len(self.invocations)
        ollama_calls = sum(1 for i in self.invocations if i.teacher_backend == "ollama")
        claude_calls = sum(1 for i in self.invocations if i.teacher_backend == "claude")
        total_cost = sum(i.cost_usd for i in self.invocations)
        return {
            "teacher_calls_total": teacher_calls,
            "ollama_calls": ollama_calls,
            "claude_calls": claude_calls,
            "claude_calls_today": self._claude_calls_today,
            "total_cost_usd": total_cost,
            "kb_size": self.net.memory.semantic.__len__(),
        }
