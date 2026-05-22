"""Multi-sentence response composition.

LLMs string together multiple related facts into a paragraph by chance
of next-token prediction; RCK does it by explicit composition. Given a
focus entity, we retrieve a small bundle of HIGH-CONFIDENCE facts about
that entity and render them as separate sentences with smooth NL flow.

Usage:
  describe(kb, "elephant") -> "Elephants have tusks. Elephants are
  mammals. The elephant has a trunk."
"""
from __future__ import annotations

from rck.knowledge_base import ShardedKnowledgeBase
from rck.nlg import render


# Relations we consider "describable" -- they tell you something useful
# about an entity when listed together.
DESCRIBE_RELATIONS = [
    "isa", "color", "size", "has", "locatedin", "lives_in",
    "madeof", "usedfor", "field", "capital", "continent",
    "version", "uses", "supports",
    "wrote", "author",
    "country", "atomic_number", "symbol",
    "orbits", "position", "height",
    "spoken_in", "works_at", "family",
    "equipment", "venue", "travels_on",
    "category", "near",
]


def describe(kb: ShardedKnowledgeBase, entity: str,
             min_conf: float = 0.15, max_sentences: int = 6) -> str:
    """Produce a multi-sentence description of `entity` from stored facts."""
    entity = entity.lower()
    parts: list[str] = []
    seen_relations: set[str] = set()
    for relation in DESCRIBE_RELATIONS:
        results = kb.query({"S": entity, "R": relation}, "O", top_k=3)
        for sym, score in results:
            if score < min_conf:
                continue
            obj = str(sym)
            sentence = render(entity, relation, obj)
            if sentence not in parts:
                parts.append(sentence)
                seen_relations.add(relation)
                break  # one fact per relation in the summary
        if len(parts) >= max_sentences:
            break
    if not parts:
        return f"I don't have any facts about {entity}."
    return " ".join(parts)


def describe_via_chain(kb: ShardedKnowledgeBase, entity: str,
                      max_steps: int = 3) -> str:
    """Describe by following an `isa` chain upward and adding inherited facts."""
    from rck.inference import _ancestors
    parts: list[str] = []
    direct = describe(kb, entity, max_sentences=3)
    parts.append(direct)
    ancestors = _ancestors(kb, entity, max_depth=max_steps)
    for ancestor in ancestors[:2]:
        ancestor_desc = describe(kb, ancestor, max_sentences=2)
        if "I don't have" not in ancestor_desc:
            parts.append(f"As a {ancestor}, " + ancestor_desc.lower())
    return " ".join(parts)
