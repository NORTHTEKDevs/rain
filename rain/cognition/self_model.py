# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/self_model.py -- adapted from RCK 4K FHRR to bipolar 10K.
"""Structural self-model: a small KB of facts about RAIN itself."""

from __future__ import annotations
from rain.core.knowledge_base import ShardedKB


DEFAULT_SELF_FACTS: list[tuple[str, str, str]] = [
    ("rain", "is_a", "generative_ai"),
    ("rain", "category", "non_llm"),
    ("rain", "substrate", "vsa_bipolar_10k"),
    ("rain", "learning", "local_rules_no_backprop"),
    ("rain", "calibrated", "yes"),
    ("rain", "continual_learning", "yes"),
    ("rain", "shows_work", "yes"),
    ("rain", "models_other_minds", "yes"),
    ("rain", "creator", "kristian_baer"),
    ("rain", "organization", "northtek"),
]


class SelfModel:
    """Stores facts about RAIN itself in a dedicated sharded HRR KB."""

    def __init__(self, dim: int = 10000, num_shards: int = 8, seed: int = 0) -> None:
        self.kb = ShardedKB(num_shards=num_shards, dim=dim, seed=seed)
        for s, r, o in DEFAULT_SELF_FACTS:
            self.kb.write(s, r, o)

    def add_fact(self, subject: str, relation: str, object_: str) -> None:
        self.kb.write(subject, relation, object_)

    def query(self, subject: str, relation: str) -> str | None:
        return self.kb.query(subject, relation)

    def describe(self) -> list[str]:
        """Return a list of natural-language sentences from stored self-facts."""
        sentences = []
        for s, r, o in DEFAULT_SELF_FACTS:
            current = self.kb.query(s, r)
            if current:
                sentences.append(f"{s} {r.replace('_', ' ')} {current.replace('_', ' ')}")
        return sentences
