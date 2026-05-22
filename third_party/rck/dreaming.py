"""Dreaming / consolidation -- the analog of biological sleep cycles.

When the agent is idle, run consolidation passes that:

  1. Promote recurring episodic events into semantic memory.
     ("user asked about France's capital 5 times" → store as preference)
  2. Detect contradictions between newly-ingested and existing facts.
     ("the new doc says sky is green; existing fact says blue")
  3. Compress near-duplicate facts.
     ("dog has fur" + "dogs have fur" → keep one, mark synonyms)
  4. Decay low-confidence facts that haven't been reinforced.
     (provenance.decay_all + low_confidence_facts → forget)
  5. Generate new abstractions by clustering facts with same shape.
     ("17 mammals have fur" → store (mammal, has, fur) as generalisation)

Each pass is bounded; consolidation runs in O(N) over the episodic /
provenance stores. Suitable for background invocation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase
from rck.memory_hierarchy import EpisodicMemory, consolidate_episodic_to_semantic
from rck.provenance import ProvenanceStore


@dataclass
class ConsolidationReport:
    """Diagnostic dump of one consolidation pass."""
    promotions: list[tuple[str, int]] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    duplicates_merged: list[tuple[str, str, str]] = field(default_factory=list)
    decayed_facts: list[tuple[str, str, str]] = field(default_factory=list)
    abstractions: list[tuple[str, str, str]] = field(default_factory=list)
    total_changes: int = 0


def detect_contradictions(kb: ShardedKnowledgeBase,
                          provenance: ProvenanceStore,
                          *, threshold: float = 0.20) -> list[dict]:
    """Find (S, R) pairs where two distinct O values both have high
    confidence. These are contradictions worth flagging."""
    pairs: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for shard in kb._shards:
        for fact in shard._facts:
            s = str(fact.get("S", "")); r = str(fact.get("R", ""))
            o = str(fact.get("O", ""))
            pairs[(s, r)].append((o, 1.0))  # presence-based; refine via provenance
    contradictions: list[dict] = []
    for (s, r), values in pairs.items():
        if len(values) < 2:
            continue
        # If R is single-valued (color, capital, isa for non-hierarchical
        # entities) AND we have multiple distinct O values, flag.
        SINGLE_VALUED = {"capital", "color", "version", "atomic_number",
                         "symbol", "founded_year", "year"}
        if r in SINGLE_VALUED:
            distinct = set(o for o, _ in values)
            if len(distinct) >= 2:
                contradictions.append({
                    "subject": s, "relation": r,
                    "competing_values": sorted(distinct),
                })
    return contradictions


def compress_duplicates(kb: ShardedKnowledgeBase) -> list[tuple[str, str, str]]:
    """Find exact duplicate facts (S, R, O all match) stored more than
    once. We don't actually re-store; HRR memory is already idempotent
    by bipolar binding. But we can dedupe the _facts list."""
    removed: list[tuple[str, str, str]] = []
    for shard in kb._shards:
        seen: set[tuple[str, str, str]] = set()
        keep: list[dict] = []
        for fact in shard._facts:
            key = (str(fact.get("S", "")),
                   str(fact.get("R", "")),
                   str(fact.get("O", "")))
            if key in seen:
                removed.append(key)
                continue
            seen.add(key)
            keep.append(fact)
        shard._facts = keep
    return removed


def generate_abstractions(kb: ShardedKnowledgeBase,
                          *, min_support: int = 5) -> list[tuple[str, str, str]]:
    """Find (parent, R, O) patterns where >=min_support children share
    the same R, O. Promote to a generalisation."""
    # parent -> list of (child, r, o)
    child_facts_by_parent: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # First, find all parent-child relations.
    parents_of: dict[str, list[str]] = defaultdict(list)
    for shard in kb._shards:
        for fact in shard._facts:
            r = str(fact.get("R", ""))
            if r in ("isa", "kind", "category"):
                parents_of[str(fact.get("S", ""))].append(str(fact.get("O", "")))
    for shard in kb._shards:
        for fact in shard._facts:
            s = str(fact.get("S", ""))
            r = str(fact.get("R", ""))
            o = str(fact.get("O", ""))
            if r in ("isa", "kind", "category"):
                continue
            for parent in parents_of.get(s, []):
                child_facts_by_parent[parent].append((s, r, o))

    abstractions: list[tuple[str, str, str]] = []
    for parent, items in child_facts_by_parent.items():
        # Count (r, o) frequencies among children.
        ro_counts: Counter[tuple[str, str]] = Counter()
        for child, r, o in items:
            ro_counts[(r, o)] += 1
        for (r, o), count in ro_counts.items():
            if count >= min_support:
                # Already exists?
                existing, score = kb.answer({"S": parent, "R": r}, "O")
                if str(existing) != o or score < 0.10:
                    abstractions.append((parent, r, o))
                    kb.store({"S": parent, "R": r, "O": o})
    return abstractions


def decay_and_prune(provenance: ProvenanceStore, kb: ShardedKnowledgeBase,
                    *, decay_factor: float = 0.995,
                    forget_threshold: float = 0.05) -> list[tuple[str, str, str]]:
    """Apply gentle confidence decay; forget facts that fall below the
    threshold."""
    provenance.decay_all(factor=decay_factor)
    to_forget = provenance.low_confidence_facts(threshold=forget_threshold)
    for s, r, o in to_forget:
        kb.forget({"S": s, "R": r, "O": o})
        provenance.forget(s, r, o)
    return to_forget


def consolidate(
    kb: ShardedKnowledgeBase,
    episodic: EpisodicMemory,
    provenance: ProvenanceStore,
    *, abstraction_support: int = 5,
    decay_factor: float = 0.999,
    forget_threshold: float = 0.02,
) -> ConsolidationReport:
    """Run one full consolidation pass. Suitable to call periodically
    (every N turns, or during user-idle time)."""
    report = ConsolidationReport()

    # 1. Episodic promotions.
    patterns = consolidate_episodic_to_semantic(episodic, threshold=3)
    report.promotions = patterns

    # 2. Contradictions.
    report.contradictions = detect_contradictions(kb, provenance)

    # 3. Duplicates.
    report.duplicates_merged = compress_duplicates(kb)

    # 4. Abstractions.
    report.abstractions = generate_abstractions(
        kb, min_support=abstraction_support,
    )

    # 5. Decay.
    report.decayed_facts = decay_and_prune(
        provenance, kb,
        decay_factor=decay_factor, forget_threshold=forget_threshold,
    )

    report.total_changes = (
        len(report.promotions) + len(report.contradictions)
        + len(report.duplicates_merged) + len(report.abstractions)
        + len(report.decayed_facts)
    )
    return report
