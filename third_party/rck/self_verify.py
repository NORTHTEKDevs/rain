"""Self-verification loop -- a second pass before answering the user.

When the agent has produced an answer via the structured path, it can
ASK ITSELF a verification question and check whether the same fact
comes back. If yes -> ship the answer. If no -> downgrade confidence
or refuse.

Three verification modes:
  * **roundtrip**: given (S, R, O), query (S, R, ?) and verify O is top-K.
  * **reverse**: query (?, R, O) and verify S is top-K.
  * **sibling_sanity**: check sibling entities give CONSISTENT answers
    for the same relation -- e.g. if "dog has fur" but no other mammal
    has fur, that's suspicious.

This is RCK's analog of LLM "self-consistency" prompting, but
architectural rather than prompt-engineered. The verification is
guaranteed to be deterministic because both passes use the same HRR.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class VerificationResult:
    verified: bool
    confidence_in: float
    confidence_out: float
    method: str
    notes: str = ""


def verify_roundtrip(kb: ShardedKnowledgeBase,
                     s: str, r: str, o: str,
                     *, top_k: int = 5) -> VerificationResult:
    """Re-query (S, R, ?) and check O appears in top-K."""
    candidates = kb.query({"S": s.lower(), "R": r.lower()}, "O", top_k=top_k)
    top_syms = [(str(sym), float(c)) for sym, c in candidates]
    o_l = o.lower()
    rank = next((i for i, (sym, _) in enumerate(top_syms) if sym == o_l), None)
    if rank is None:
        return VerificationResult(
            verified=False, confidence_in=0.0, confidence_out=0.0,
            method="roundtrip",
            notes=f"{o!r} not in top-{top_k} for ({s}, {r}, ?)",
        )
    conf = top_syms[rank][1]
    # Discount by rank: top-1 = full confidence; top-K = decayed.
    confidence_out = conf * (1.0 / (1.0 + rank))
    return VerificationResult(
        verified=conf > 0.05,
        confidence_in=conf,
        confidence_out=confidence_out,
        method="roundtrip",
        notes=f"rank={rank}, cosine={conf:.3f}",
    )


def verify_reverse(kb: ShardedKnowledgeBase,
                   s: str, r: str, o: str,
                   *, top_k: int = 5) -> VerificationResult:
    """Re-query (?, R, O) and check S appears in top-K."""
    candidates = kb.query({"R": r.lower(), "O": o.lower()}, "S", top_k=top_k)
    top_syms = [(str(sym), float(c)) for sym, c in candidates]
    s_l = s.lower()
    rank = next((i for i, (sym, _) in enumerate(top_syms) if sym == s_l), None)
    if rank is None:
        return VerificationResult(
            verified=False, confidence_in=0.0, confidence_out=0.0,
            method="reverse",
            notes=f"{s!r} not in top-{top_k} for (?, {r}, {o})",
        )
    conf = top_syms[rank][1]
    confidence_out = conf * (1.0 / (1.0 + rank))
    return VerificationResult(
        verified=conf > 0.05,
        confidence_in=conf,
        confidence_out=confidence_out,
        method="reverse",
        notes=f"rank={rank}, cosine={conf:.3f}",
    )


def verify_sibling_sanity(kb: ShardedKnowledgeBase,
                          s: str, r: str, o: str) -> VerificationResult:
    """If `r` is a property typically shared by siblings (has, color,
    locatedin), check that >=1 sibling has the same property.

    Sibling = entity sharing the same isa parent.
    """
    parent_results = kb.query({"S": s.lower(), "R": "isa"}, "O", top_k=1)
    if not parent_results or parent_results[0][1] < 0.10:
        return VerificationResult(
            verified=True,  # no siblings to check; abstain
            confidence_in=0.5, confidence_out=0.5,
            method="sibling_sanity",
            notes="no isa parent found",
        )
    parent = str(parent_results[0][0])
    siblings = kb.query({"R": "isa", "O": parent}, "S", top_k=20)
    siblings = [str(sym) for sym, c in siblings if c >= 0.10 and str(sym) != s.lower()]
    if not siblings:
        return VerificationResult(
            verified=True, confidence_in=0.5, confidence_out=0.5,
            method="sibling_sanity",
            notes="no siblings to compare against",
        )
    # Count siblings with the same (R, O).
    same = 0
    for sib in siblings[:10]:
        ans, score = kb.answer({"S": sib, "R": r.lower()}, "O")
        if ans is not None and str(ans) == o.lower() and score > 0.10:
            same += 1
    fraction = same / max(1, min(len(siblings), 10))
    return VerificationResult(
        verified=fraction >= 0.20,
        confidence_in=fraction,
        confidence_out=fraction,
        method="sibling_sanity",
        notes=f"{same}/{min(len(siblings), 10)} siblings share this property",
    )


def verify(kb: ShardedKnowledgeBase, s: str, r: str, o: str,
           *, methods: list[str] | None = None) -> dict:
    """Run all (or selected) verification methods. Return aggregate."""
    methods = methods or ["roundtrip", "reverse"]
    results: dict[str, VerificationResult] = {}
    if "roundtrip" in methods:
        results["roundtrip"] = verify_roundtrip(kb, s, r, o)
    if "reverse" in methods:
        results["reverse"] = verify_reverse(kb, s, r, o)
    if "sibling_sanity" in methods:
        results["sibling_sanity"] = verify_sibling_sanity(kb, s, r, o)
    verified_count = sum(1 for r_ in results.values() if r_.verified)
    avg_conf = sum(r_.confidence_out for r_ in results.values()) / max(1, len(results))
    return {
        "verified": verified_count == len(results),
        "verified_partial": verified_count >= 1,
        "verification_count": verified_count,
        "total_methods": len(results),
        "average_confidence": avg_conf,
        "by_method": {name: {
            "verified": r.verified, "confidence": r.confidence_out,
            "notes": r.notes,
        } for name, r in results.items()},
    }
