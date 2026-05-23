# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/dialogue.py — adapted from RCK FHRR.
"""DialogueContext tracks the running conversation state.

Tracks last-mentioned entity and relation so that questions like
'what about the grass?' or 'what color is it?' can be resolved by
substituting the missing slot with the last-mentioned one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DialogueContext:
    last_entity: str | None = None
    last_relation: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)  # (role, utterance)

    def remember(self, entity: str | None = None, relation: str | None = None) -> None:
        if entity is not None:
            self.last_entity = entity
        if relation is not None:
            self.last_relation = relation

    def record_turn(self, role: str, utterance: str) -> None:
        self.history.append((role, utterance))

    def resolve_references(self, entity: str | None, relation: str | None) -> tuple[str | None, str | None]:
        """If entity or relation is None (or pronoun-like), substitute from context."""
        resolved_entity = entity if entity else self.last_entity
        resolved_relation = relation if relation else self.last_relation
        return resolved_entity, resolved_relation

    def with_default_topic(self, partial_question: str) -> str:
        """Treat 'what about X?' style prompts by appending the last entity/relation context."""
        if self.last_relation and self.last_entity:
            return f"{partial_question} ({self.last_entity}.{self.last_relation})"
        return partial_question

    def reset(self) -> None:
        self.last_entity = None
        self.last_relation = None
        self.history.clear()
