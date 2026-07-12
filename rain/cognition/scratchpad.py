"""Reasoning scratchpad -- explicit chain-of-thought for HYMN-Plus v2.

LLMs do CoT implicitly by generating tokens that prompt themselves toward
the right answer ("Let me think step by step..."). RAIN can do it
explicitly using structured KB writes:

  1. User asks a multi-hop question: "Is amoxicillin safe with warfarin?"
  2. ScratchpadReasoner decomposes into sub-questions:
     - amoxicillin -> drug_class
     - warfarin -> drug_class
     - drug_class(amoxicillin) interacts_with drug_class(warfarin)?
  3. For each sub-question, ask the agent (KB + HYMN-Plus v2 fallback)
  4. Cache intermediate answers as scratchpad facts in the KB
  5. The model can attend to the scratchpad facts via KB-Attention
     during the final answer generation -> chain-of-thought without
     LLM-style "thinking out loud"

The scratchpad is just a temporary KB slice. Each step's answer becomes
a (scratch:s, scratch:r, scratch:o) triple that the model retrieves
via KB-attention. After the final answer, the scratchpad can be cleared
(or kept as session memory).

Scope of this prototype:
  * Sequential reasoning (one sub-question at a time)
  * Simple decomposition rules (regex-based question splitter)
  * Cache hits in ConsciousAgent.kb so attention picks them up

Future: learned decomposition, parallel sub-questions, scratchpad
attention as a separate KB-Attention block, ToT (Tree-of-Thoughts)
expansion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    """One sub-question + the agent's answer to it."""

    sub_question: str
    sub_subject: str
    sub_relation: str
    answer: str
    epistemic: str  # know / think / guess / unknown
    citations: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    """The full chain that led to the final answer. Auditable."""

    original_question: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    final_epistemic: str = "unknown"


class ScratchpadReasoner:
    """Drives multi-step reasoning over a ConsciousAgent.

    Usage:
        reasoner = ScratchpadReasoner(agent)
        trace = reasoner.reason("does the lion eat fish?")
        print(trace.final_answer)
        for step in trace.steps:
            print("  ->", step.sub_question, "::", step.answer)
    """

    def __init__(self, agent, max_steps: int = 5) -> None:
        self.agent = agent
        self.max_steps = max_steps

    # Very simple question-decomposition rules. Real CoT will need a
    # learned decomposer or LLM judge; this is the architectural hook.
    _PATTERNS = [
        # "Is X safe with Y?" -> ask X.interacts_with(Y)
        (
            re.compile(r"is\s+(\w+)\s+safe\s+with\s+(\w+)", re.IGNORECASE),
            lambda m: [(m.group(1), "interacts_with", m.group(2))],
        ),
        # "Does X eat Y?" -> ask X.eats(Y) AND X.diet
        (
            re.compile(r"does\s+(?:the\s+)?(\w+)\s+eat\s+(\w+)", re.IGNORECASE),
            lambda m: [(m.group(1), "eats", m.group(2)), (m.group(1), "diet", "")],
        ),
        # "Where does X live?" -> X.lives_in
        (
            re.compile(r"where\s+does\s+(?:the\s+)?(\w+)\s+live", re.IGNORECASE),
            lambda m: [(m.group(1), "lives_in", "")],
        ),
        # "What is the capital of X?" -> X.capital
        (
            re.compile(r"what\s+is\s+the\s+capital\s+of\s+(\w+)", re.IGNORECASE),
            lambda m: [(m.group(1), "capital", "")],
        ),
        # "What color is X?" -> X.color
        (
            re.compile(r"what\s+color\s+is\s+(?:the\s+)?(\w+)", re.IGNORECASE),
            lambda m: [(m.group(1), "color", "")],
        ),
    ]

    def decompose(self, question: str) -> list[tuple[str, str, str]]:
        """Return a list of (subject, relation, optional-target-object) sub-queries."""
        question = question.strip().rstrip("?.! ")
        for pat, builder in self._PATTERNS:
            m = pat.search(question)
            if m:
                return builder(m)
        # Fallback: try to split on whitespace and treat as (subject, relation)
        parts = question.split()
        if len(parts) >= 2:
            return [(parts[-1], "_".join(parts[:-1]).lower(), "")]
        return [(question, "isa", "")]

    def reason(self, question: str) -> ReasoningTrace:
        trace = ReasoningTrace(original_question=question)
        subq = self.decompose(question)
        for s, r, _target in subq[: self.max_steps]:
            ans = self.agent.ask(s, r)
            trace.steps.append(
                ReasoningStep(
                    sub_question=f"{s}.{r}",
                    sub_subject=s,
                    sub_relation=r,
                    answer=ans.text,
                    epistemic=ans.epistemic,
                    citations=list(ans.citations),
                )
            )
            # Cache intermediate answer as scratchpad triple so future
            # KB-attention queries can attend to it. This is the key
            # mechanism: previous reasoning steps become retrievable
            # facts for the next step.
            stored_obj = ans.citations[0][2] if ans.citations else None
            if stored_obj:
                self.agent.kb.write(f"scratch:{s}", f"scratch:{r}", stored_obj)

        if trace.steps:
            # Synthesize the final answer from the steps. v0: concatenate.
            best_step = max(
                trace.steps,
                key=lambda s: 1.0 if s.epistemic in ("know", "think") else 0.0,
            )
            trace.final_answer = best_step.answer
            trace.final_epistemic = best_step.epistemic
        else:
            trace.final_answer = "I don't have enough context to reason about that."
            trace.final_epistemic = "unknown"
        return trace
