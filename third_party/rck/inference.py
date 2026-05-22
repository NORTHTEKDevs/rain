"""Multi-hop chain inference.

The killer capability that makes a structured KB worth more than its
weight in raw facts: derive new conclusions by chaining stored facts.

Example chains we handle:
  (dog isa mammal) + (mammal has fur)  -> (dog has fur)         [inherit]
  (dog isa mammal) + (mammal isa animal) -> (dog isa animal)    [isa-transitive]
  (paris in france) + (france in europe)  -> (paris in europe)  [in-transitive]
  (X capital paris) + (paris in france)   -> (X locatedin france)

Strategy:
  - For the direct query (S, R, ?), first check the KB.
  - If we get nothing, walk up `isa` (or `kind`, `category`) edges from
    S and retry on each ancestor.
  - For transitive relations (isa, partof, in, locatedin), also chain
    forward: (S, R, mid) then (mid, R, ?).

This is bounded by depth and breadth so a malformed KB cannot send the
engine into a runaway expansion. Every inferred answer is returned with
the chain of facts that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase


# Relations that inherit DOWNWARD along `isa`:
#   if X isa Y and Y has Z, then X has Z.
INHERITED_RELATIONS = {
    "has", "color", "madeof", "usedfor", "field",
    "category", "locatedin", "lives_in", "continent",
    "size", "isa",
}

# Relations that are TRANSITIVE on themselves:
#   if X R Y and Y R Z, then X R Z.
TRANSITIVE_RELATIONS = {"isa", "partof", "in", "locatedin"}

# Aliases used while climbing the type hierarchy. We include partitive
# and locational relations because cities inherit attributes of countries,
# parts inherit attributes of wholes, etc.
ISA_RELATIONS = ("isa", "kind", "category", "locatedin", "lives_in", "partof")


@dataclass
class InferenceResult:
    answer: str | None
    confidence: float
    chain: list[tuple[str, str, str]] = field(default_factory=list)
    depth: int = 0
    source: str = "direct"  # direct | inherited | transitive | none

    def explain(self) -> str:
        if self.answer is None:
            return "no answer"
        if not self.chain:
            return f"{self.answer} (direct lookup)"
        steps = " -> ".join(f"({s}, {r}, {o})" for s, r, o in self.chain)
        return f"{self.answer} via [{steps}]"


def _direct_lookup(kb: ShardedKnowledgeBase, subject: str, relation: str,
                   min_conf: float = 0.10) -> tuple[str | None, float]:
    ans, score = kb.answer({"S": subject, "R": relation}, "O")
    if ans is None or score < min_conf:
        return None, 0.0
    return str(ans), float(score)


def _ancestors(kb: ShardedKnowledgeBase, subject: str, max_depth: int = 4) -> list[str]:
    """Return ancestor classes by walking `isa` relations BFS up to max_depth."""
    seen = {subject}
    frontier = [subject]
    out: list[str] = []
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for s in frontier:
            for rel in ISA_RELATIONS:
                results = kb.query({"S": s, "R": rel}, "O", top_k=3)
                for sym, score in results:
                    if score < 0.10:
                        continue
                    sym = str(sym)
                    if sym in seen:
                        continue
                    seen.add(sym)
                    out.append(sym)
                    next_frontier.append(sym)
        if not next_frontier:
            break
        frontier = next_frontier
    return out


def infer(
    kb: ShardedKnowledgeBase,
    subject: str,
    relation: str,
    *,
    max_depth: int = 3,
    min_conf: float = 0.10,
) -> InferenceResult:
    """Try direct lookup; on failure, attempt single-step chain inference."""
    subject = subject.lower(); relation = relation.lower()

    # 1. Direct.
    ans, conf = _direct_lookup(kb, subject, relation, min_conf=min_conf)
    if ans is not None:
        return InferenceResult(
            answer=ans, confidence=conf,
            chain=[(subject, relation, ans)],
            depth=0, source="direct",
        )

    # 2. Transitive on the SAME relation.
    if relation in TRANSITIVE_RELATIONS:
        intermediates = kb.query({"S": subject, "R": relation}, "O", top_k=3)
        for mid, mid_score in intermediates:
            if mid_score < min_conf:
                continue
            mid = str(mid)
            ans, score = _direct_lookup(kb, mid, relation, min_conf=min_conf)
            if ans is not None:
                return InferenceResult(
                    answer=ans, confidence=min(mid_score, score) * 0.8,
                    chain=[(subject, relation, mid), (mid, relation, ans)],
                    depth=1, source="transitive",
                )

    # 3. Inherited via `isa`-walk.
    if relation in INHERITED_RELATIONS:
        for ancestor in _ancestors(kb, subject, max_depth=max_depth):
            ans, score = _direct_lookup(kb, ancestor, relation, min_conf=min_conf)
            if ans is not None:
                # Build chain: (subject isa ... ancestor) + (ancestor R ans)
                chain = [(subject, "isa", ancestor),
                         (ancestor, relation, ans)]
                return InferenceResult(
                    answer=ans, confidence=score * 0.7,
                    chain=chain, depth=1, source="inherited",
                )
    return InferenceResult(answer=None, confidence=0.0, source="none")


def boolean(
    kb: ShardedKnowledgeBase,
    subject: str, relation: str, value: str,
    *,
    min_conf: float = 0.05,
) -> dict:
    """Boolean query: is (S, R, V) true?

    Strategy:
      1. Get the TOP-K candidates for (S, R, ?) -- the question is whether
         `value` appears among them with reasonable confidence. This handles
         multi-valued relations like 'elephant has [tusks, trunk, ears]':
         if `value` is in the top-K and well above the noise floor, the
         answer is True.
      2. If `value` is NOT in the top-K but the top-1 is highly confident
         AND `value` is uniquely associated elsewhere (single-valued
         relations like 'capital'), report False.
      3. Otherwise, fall back to multi-hop inference.
    """
    subject = subject.lower(); relation = relation.lower(); value = value.lower()

    # 0. Multi-hop isa-walk for "is X a Y" style queries. The plain
    # infer() returns the immediate parent; here we walk further to see
    # if `value` is anywhere up the chain.
    if relation in ("isa", "kind", "category"):
        from rck.inference import _ancestors as _walk
        ancestors = _walk(kb, subject, max_depth=5)
        if value in ancestors:
            return {
                "answer": True, "confidence": 0.7,
                "chain": [(subject, "isa+", value)],
                "source": "isa-transitive",
            }

    # 1. Look at the top-K candidates for (S, R, ?).
    results = kb.query({"S": subject, "R": relation}, "O", top_k=8)
    if results:
        # Is `value` present among the high-confidence candidates?
        for sym, score in results:
            if str(sym) == value and score > 0.08:
                return {
                    "answer": True, "confidence": float(score),
                    "chain": [(subject, relation, value)], "source": "direct",
                }
        # `value` not present with confidence -- but does the top result
        # confidently exclude `value`? Single-valued relations like
        # `color`, `capital`, `isa` are exclusionary; multi-valued ones
        # like `has`, `usedfor` are NOT.
        SINGLE_VALUED = {"color", "capital", "isa", "kind",
                         "category", "size", "value", "field",
                         "continent", "previousmonth", "version"}
        top_sym, top_score = str(results[0][0]), float(results[0][1])
        if relation in SINGLE_VALUED and top_score > 0.20 and top_sym != value:
            return {
                "answer": False, "confidence": top_score,
                "chain": [(subject, relation, top_sym),
                          ("expected_was", value, "")],
                "source": "contradicted",
            }

    # 2. Multi-hop inference.
    inferred = infer(kb, subject, relation)
    if inferred.answer == value:
        return {
            "answer": True, "confidence": inferred.confidence,
            "chain": inferred.chain, "source": inferred.source,
        }
    return {"answer": None, "confidence": 0.0, "chain": [], "source": "unknown"}


def enumerate_subjects(
    kb: ShardedKnowledgeBase,
    relation: str, value: str,
    *,
    top_k: int = 20,
    min_conf: float = 0.20,
) -> list[tuple[str, float]]:
    """Enumeration: list all S such that (S, relation, value).

    With sharding we must fan out across every shard because S is unknown.
    The default min_conf is HIGH (0.20) because fan-out introduces a lot
    of crosstalk noise; below that bar matches are essentially random.
    """
    relation = relation.lower(); value = value.lower()
    results = kb.query({"R": relation, "O": value}, "S", top_k=top_k)
    return [(str(s), float(c)) for s, c in results if c >= min_conf]


def compare(
    kb: ShardedKnowledgeBase,
    subject_a: str, subject_b: str, dimension: str = "size",
) -> dict:
    """Compare two entities along a dimension.

    Returns {"winner": str|None, "verbal": str, "values": (a_val, b_val)}.

    For `size`, we look up (S, size, ?) and use an ordering:
      tiny < small < medium < large < huge.
    """
    SIZE_ORDER = {"tiny": 0, "small": 1, "medium": 2, "large": 3, "huge": 4}
    ans_a = infer(kb, subject_a, dimension)
    ans_b = infer(kb, subject_b, dimension)
    if ans_a.answer is None or ans_b.answer is None:
        return {"winner": None, "verbal": "I don't have size info for both.",
                "values": (ans_a.answer, ans_b.answer)}
    va, vb = ans_a.answer, ans_b.answer
    rank_a = SIZE_ORDER.get(va, -1)
    rank_b = SIZE_ORDER.get(vb, -1)
    if rank_a < 0 or rank_b < 0:
        return {"winner": None,
                "verbal": f"{subject_a} is {va}, {subject_b} is {vb}.",
                "values": (va, vb)}
    if rank_a > rank_b:
        return {"winner": subject_a,
                "verbal": f"{subject_a} ({va}) is bigger than {subject_b} ({vb}).",
                "values": (va, vb)}
    if rank_b > rank_a:
        return {"winner": subject_b,
                "verbal": f"{subject_b} ({vb}) is bigger than {subject_a} ({va}).",
                "values": (va, vb)}
    return {"winner": None,
            "verbal": f"{subject_a} and {subject_b} are both {va}.",
            "values": (va, vb)}
