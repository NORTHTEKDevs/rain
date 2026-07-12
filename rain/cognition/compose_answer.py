"""describe(entity) -> multi-sentence paragraph from stored facts."""

from __future__ import annotations

from rain.cognition.nlg import render
from rain.core.knowledge_base import ShardedKB


def describe(kb: ShardedKB, entity: str, relations: list[str]) -> str:
    """Build a paragraph from all known (entity, relation, object) triples."""
    sentences: list[str] = []
    for r in relations:
        o = kb.query(entity, r)
        if o:
            sentences.append(render(entity, r, o))
    if not sentences:
        return f"I don't know anything about {entity} yet."
    return " ".join(sentences)
