# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/conscious_agent.py — adapted to use RAIN's
# bipolar 10K modules.
"""ConsciousAgent — top-level orchestrator.

Composes Codebook + ShardedKB + DialogueContext + Introspector + SelfModel +
TheoryOfMind + CalibrationTally + multi-hop inference + NL output. Exposes
the public interaction methods: ask, tell, describe, think_aloud, refute,
introspect.

This is the v0 cognitive surface. It does NOT use HYMN/LSM/FEP/Tsetlin yet
for token generation (those come online when the full EFE decoder integrates
in Phase 6). v0 ConsciousAgent operates over the symbolic-KB layer and
provides the ToM/self-model/calibration/transparency capabilities that
distinguish RAIN from LLMs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from rain.core.relational import Codebook
from rain.core.knowledge_base import ShardedKB
from rain.cognition.dialogue import DialogueContext
from rain.cognition.introspect import Introspector
from rain.cognition.self_model import SelfModel
from rain.cognition.theory_of_mind import TheoryOfMind
from rain.cognition.metacog import CalibrationTally, classify_epistemic
from rain.cognition.inference import infer, InferenceResult
from rain.cognition.explain import explain
from rain.cognition.think_aloud import narrate
from rain.cognition.compose_answer import describe as describe_fn
from rain.cognition.nlg import render as nlg_render


@dataclass
class Answer:
    """One agent reply with all the metadata RAIN exposes (calibration + epistemic + citations)."""
    text: str
    epistemic: str   # know / think / guess / unknown
    citations: list[tuple[str, str, str]] = field(default_factory=list)
    confidence: float = 0.0
    inference_source: str | None = None  # direct / inherited / transitive / None


class ConsciousAgent:
    def __init__(self, dim: int = 10000, num_shards: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.codebook = Codebook(vocab_size=4096, dim=dim, seed=seed)
        self.kb = ShardedKB(num_shards=num_shards, dim=dim, seed=seed)
        self.dialogue = DialogueContext()
        self.introspect = Introspector(capacity=128)
        self.self_model = SelfModel(dim=dim, num_shards=8, seed=seed + 1)
        self.tom = TheoryOfMind(dim=dim, num_shards=8, seed=seed + 2)
        self.calibration = CalibrationTally()

    def tell(self, subject: str, relation: str, object_: str) -> None:
        """Teach a new fact. Records introspection + updates dialogue context."""
        self.kb.write(subject, relation, object_)
        self.dialogue.remember(entity=subject, relation=relation)
        self.introspect.record("learn", fact=f"{subject}.{relation}={object_}")

    def ask(self, subject: str, relation: str, think_aloud: bool = False) -> Answer:
        """Answer a fact-shaped question. Returns an Answer with epistemic class + citations."""
        # Resolve references from dialogue context
        subject_r, relation_r = self.dialogue.resolve_references(subject, relation)
        if subject_r is None or relation_r is None:
            ans = Answer(text="I don't have enough context to know what you're asking.",
                         epistemic="unknown")
            self.introspect.record("refuse", reason="missing context")
            return ans

        result = infer(self.kb, subject_r, relation_r)
        # Compute confidence: 1.0 for direct, 0.7 for inherited, 0.5 for transitive, 0.0 for None
        conf_map = {"direct": 1.0, "inherited": 0.7, "transitive": 0.5, None: 0.0}
        confidence = conf_map[result.source]
        cal = self.calibration.calibration(relation_r)
        epistemic = classify_epistemic(confidence, relation_calibration=cal)

        if result.answer is None:
            text = "I don't know that yet."
            self.introspect.record("refuse", reason="no fact / no chain")
        elif think_aloud:
            text = narrate(f"what is {relation_r} of {subject_r}?", result)
            self.introspect.record("explain", topic=f"{subject_r}.{relation_r}")
        else:
            explanation = explain(result)
            text = explanation.text
            self.introspect.record("answer", answer=result.answer)

        # Update dialogue context with this turn's subject + relation
        self.dialogue.remember(entity=subject_r, relation=relation_r)
        self.dialogue.record_turn("user", f"ask({subject_r}, {relation_r})")
        self.dialogue.record_turn("rain", text)

        return Answer(
            text=text,
            epistemic=epistemic,
            citations=list(result.chain),
            confidence=confidence,
            inference_source=result.source,
        )

    def describe(self, entity: str, relations: list[str] | None = None) -> str:
        """Multi-sentence description from stored facts."""
        if relations is None:
            relations = [
                "isa", "category", "color", "size", "capital_of", "locatedin",
                "lives_in", "made_of", "has_part", "wrote", "creator",
            ]
        out = describe_fn(self.kb, entity, relations)
        self.introspect.record("explain", topic=entity)
        return out

    def feedback(self, relation: str, was_correct: bool) -> None:
        """User feedback on a previous answer — updates per-relation calibration tally."""
        self.calibration.update(relation, was_correct)

    def self_describe(self) -> str:
        """Plain-language description of RAIN itself from self-model."""
        return " ".join(self.self_model.describe())

    def what_just_happened(self) -> str:
        """Introspection narration."""
        return self.introspect.narrate()
