# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/inference.py — adapted from RCK FHRR.
"""Multi-hop chain inference over a ShardedKB.

infer(kb, subject, relation) returns:
  - direct: the stored answer if (subject, relation) is in KB.
  - inherited: walks parent relations (isa, kind, category, partof, locatedin)
    upward and queries each ancestor for the same relation; returns the first
    hit along with the inheritance chain.
  - transitive: chains the same relation transitively when supported (e.g.
    locatedin -> locatedin -> ...).

Returns an InferenceResult with .answer, .chain, .source ("direct" / "inherited"
/ "transitive" / None).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rain.core.knowledge_base import ShardedKB

# Parent relations that propagate properties from ancestors to descendants
PARENT_RELATIONS: tuple[str, ...] = (
    "isa",
    "kind",
    "category",
    "partof",
    "locatedin",
    "lives_in",
)


@dataclass
class InferenceResult:
    answer: str | None
    chain: list[tuple[str, str, str]] = field(default_factory=list)
    source: str | None = None  # "direct" | "inherited" | "transitive" | None


def infer(kb: ShardedKB, subject: str, relation: str, max_depth: int = 5) -> InferenceResult:
    """Try direct first, then walk parent relations to inherit a property."""
    # Direct
    direct = kb.query(subject, relation)
    if direct:
        return InferenceResult(answer=direct, chain=[(subject, relation, direct)], source="direct")

    # Inherit: walk up via PARENT_RELATIONS, query at each step
    chain: list[tuple[str, str, str]] = []
    current = subject
    visited = {current}
    for _depth in range(max_depth):
        # Find a parent of `current`
        parent: str | None = None
        for pr in PARENT_RELATIONS:
            candidate = kb.query(current, pr)
            if candidate and candidate not in visited:
                chain.append((current, pr, candidate))
                parent = candidate
                visited.add(candidate)
                break
        if parent is None:
            break
        # Query `relation` on the parent
        ans = kb.query(parent, relation)
        if ans:
            chain.append((parent, relation, ans))
            return InferenceResult(answer=ans, chain=chain, source="inherited")
        current = parent

    # Transitive (same relation chained)
    if relation in PARENT_RELATIONS:
        chain = []
        current = subject
        visited = {current}
        for _depth in range(max_depth):
            ans = kb.query(current, relation)
            if not ans or ans in visited:
                break
            chain.append((current, relation, ans))
            visited.add(ans)
            current = ans
        if chain:
            return InferenceResult(answer=chain[-1][2], chain=chain, source="transitive")

    return InferenceResult(answer=None, source=None)
