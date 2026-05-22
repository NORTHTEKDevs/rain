"""Dialogue context manager.

Multi-turn conversations need to track:
  - The last entity mentioned (for "what about it?" / "what color is it?").
  - The last relation (for "what about the grass?" defaulting to "color").
  - Recent question/answer history (for retrieval explanations).

This module is small + explicit: no learning, just a sliding context that
the question classifier consults BEFORE running the parser. References
like "it", "that", "the same" are resolved by substitution.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rck.tokenizer import tokenize


# Words that indicate the entity should be inherited from the previous turn.
PRONOUNS = {"it", "they", "them", "that", "this", "those", "these"}


@dataclass
class DialogueContext:
    """Sliding context across conversational turns."""

    max_turns: int = 16
    last_entity: str | None = field(default=None, init=False)
    last_relation: str | None = field(default=None, init=False)
    last_answer: str | None = field(default=None, init=False)
    history: deque = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.max_turns)

    def record(self, question: str, parsed: dict | None, answer: str | None) -> None:
        if parsed:
            if "entity" in parsed:
                self.last_entity = parsed["entity"]
            if "relation" in parsed:
                self.last_relation = parsed["relation"]
        if answer is not None:
            self.last_answer = answer
        self.history.append({
            "question": question, "parsed": parsed, "answer": answer,
        })

    def reset(self) -> None:
        self.last_entity = None
        self.last_relation = None
        self.last_answer = None
        self.history.clear()

    def resolve_references(self, question: str) -> str:
        """Substitute pronouns with the last entity. Conservative -- only
        rewrites if the pronoun appears AND we have a previous entity."""
        if self.last_entity is None:
            return question
        toks = tokenize(question)
        if not any(t in PRONOUNS for t in toks):
            return question
        # Replace whole-word pronouns with the last entity.
        replaced = []
        rewritten = False
        for t in toks:
            if t in PRONOUNS and not rewritten:
                replaced.append(self.last_entity)
                rewritten = True
            else:
                replaced.append(t)
        return " ".join(replaced)

    def with_default_topic(self, question: str) -> str:
        """For very short follow-ups, re-inject the last relation.

        Patterns handled:
          'what about the grass?'  -> 'what <last_rel> is the grass?'
          'what about grass?'      -> 'what <last_rel> is the grass?'
          'what about it?'         -> uses last_entity instead
          'and the grass?'         -> 'what <last_rel> is the grass?'
        """
        if self.last_relation is None:
            return question
        toks = [t for t in tokenize(question) if t not in {"?", "."}]
        if not toks:
            return question
        rest: list[str] = []
        # 'what about the X' / 'what about X'
        if toks[:2] == ["what", "about"]:
            tail = toks[2:]
            if tail and tail[0] == "the":
                tail = tail[1:]
            rest = tail
        # 'and the X'
        elif toks[:2] == ["and", "the"]:
            rest = toks[2:]
        # Whole question is just 'it' / 'them' etc.
        elif len(toks) == 1 and toks[0] in PRONOUNS and self.last_entity:
            return f"what {self.last_relation} is the {self.last_entity}?"
        else:
            return question
        # 'it' -> last_entity
        if rest and rest[0] in PRONOUNS and self.last_entity:
            rest[0] = self.last_entity
        if not rest:
            return question
        return f"what {self.last_relation} is the {' '.join(rest)}?"
