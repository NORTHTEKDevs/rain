"""RCK's self-model: structured facts about itself.

A grounded form of "self-awareness" without making any consciousness
claims. The model stores triples about itself (capabilities, limits,
architecture, current state) in the same relational substrate it uses
for any other knowledge. When asked "what are you?", "what can you do?",
"do you know X?", it retrieves answers from this self-model just like
it would for any external entity.

This implements one criterion proposed for machine self-awareness:
the model has an explicit, queryable representation of itself, and
introspective answers are grounded in retrieval rather than confabulation.
"""
from __future__ import annotations

from rck.knowledge_base import ShardedKnowledgeBase


# Canonical name the self-model uses to refer to itself.
SELF_NAME = "rck"


CORE_SELF_FACTS = [
    # Identity
    ("rck", "is", "ai"),
    ("rck", "is", "system"),
    ("rck", "name", "rck"),
    ("rck", "full_name", "resonant_cognitive_kernel"),
    ("rck", "version", "15.0.0"),
    ("rck", "kind", "neuro_symbolic"),
    ("rck", "kind", "vsa_based"),
    # Architecture
    ("rck", "uses", "vsa"),
    ("rck", "uses", "hypervectors"),
    ("rck", "uses", "predictive_coding"),
    ("rck", "uses", "liquid_state_machine"),
    ("rck", "uses", "tsetlin_machine"),
    ("rck", "uses", "global_workspace_theory"),
    ("rck", "uses", "free_energy_principle"),
    ("rck", "uses", "thousand_brains_theory"),
    ("rck", "uses", "holographic_reduced_representations"),
    # Properties
    ("rck", "runs_on", "cpu"),
    ("rck", "requires", "numpy"),
    ("rck", "does_not_use", "backpropagation"),
    ("rck", "does_not_use", "gpu"),
    ("rck", "does_not_use", "batches"),
    ("rck", "supports", "continual_learning"),
    ("rck", "supports", "one_shot_learning"),
    ("rck", "supports", "compositional_generalization"),
    ("rck", "supports", "federated_merge"),
    ("rck", "supports", "editable_knowledge"),
    ("rck", "supports", "multi_hop_inference"),
    ("rck", "supports", "boolean_questions"),
    ("rck", "supports", "enumeration_questions"),
    ("rck", "supports", "comparison_questions"),
    ("rck", "supports", "multi_turn_dialogue"),
    ("rck", "supports", "theory_of_mind"),
    ("rck", "supports", "self_awareness"),
    ("rck", "produces", "interpretable_reasoning_trace"),
    # Limits (the model knows what it cannot do)
    ("rck", "cannot", "match_gpt_fluency"),
    ("rck", "cannot", "process_images_natively"),
    ("rck", "cannot", "process_audio_natively"),
    ("rck", "cannot", "modify_its_own_architecture"),
    ("rck", "limit", "knowledge_is_finite"),
    ("rck", "limit", "no_internet_access"),
    # Behaviour
    ("rck", "behavior", "soft_reject_unknown"),
    ("rck", "behavior", "cite_confidence_scores"),
    ("rck", "behavior", "preserve_prior_knowledge"),
]


def install_self_model(kb: ShardedKnowledgeBase) -> int:
    """Insert RCK's self-knowledge into a knowledge base.

    Returns the number of facts installed.
    """
    n = 0
    for s, r, o in CORE_SELF_FACTS:
        kb.store({"S": s, "R": r, "O": o})
        n += 1
    return n


def self_describe(kb: ShardedKnowledgeBase) -> str:
    """Return a natural-language description of RCK from its own self-facts."""
    name = _first(kb, "full_name") or SELF_NAME
    version = _first(kb, "version") or "?"
    kind = _all(kb, "kind") or ["neuro-symbolic system"]
    uses = _all(kb, "uses")
    supports = _all(kb, "supports")
    cannot = _all(kb, "cannot") + _all(kb, "limit")

    lines = [
        f"I am {name} (RCK) version {version}.",
        f"I am a {', '.join(kind)}.",
    ]
    if uses:
        lines.append(f"My architecture uses: {', '.join(uses[:6])}"
                     + ("..." if len(uses) > 6 else "."))
    if supports:
        lines.append(f"I support: {', '.join(supports[:6])}"
                     + ("..." if len(supports) > 6 else "."))
    if cannot:
        lines.append(f"I cannot: {', '.join(cannot[:5])}"
                     + ("..." if len(cannot) > 5 else "."))
    return " ".join(lines)


def _first(kb: ShardedKnowledgeBase, relation: str) -> str | None:
    ans, score = kb.answer({"S": SELF_NAME, "R": relation}, "O")
    if ans is None or score < 0.10:
        return None
    return str(ans)


def _all(kb: ShardedKnowledgeBase, relation: str, top_k: int = 12,
         min_cos: float = 0.10) -> list[str]:
    """Return only HIGH-confidence matches. The default threshold filters
    out cross-binding noise from other facts that happen to live in the
    same shard."""
    results = kb.query({"S": SELF_NAME, "R": relation}, "O", top_k=top_k)
    return [str(s) for s, score in results if score >= min_cos]
