"""Federated KB merge.

Combine two agents' auxiliary state (provenance, skills, KB) into
a single target agent. Useful when two RCK agents have been trained
on disjoint domains and we want a unified one.

What this module merges:
  * HRR knowledge bases (via RelationalMemory.merge, which is HRR-
    additive: bipolar bundle sum)
  * ProvenanceStore (per-fact metadata is summed: count, last_seen
    take the max, source becomes "multi" if they disagree)
  * SkillLibrary (success / failure counters add)

What it does NOT merge:
  * QueryMemory (episode logs are typically agent-specific)
  * Chain cache (different topologies invalidate it anyway)
  * Calibration tally (per-agent metacognitive measurements)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rck.conscious_agent import ConsciousAgent


@dataclass
class MergeReport:
    skills_added: int = 0
    skills_reinforced: int = 0
    provenance_added: int = 0
    provenance_reinforced: int = 0
    kb_facts_merged: int = 0


def merge_agents(target: "ConsciousAgent",
                 source: "ConsciousAgent") -> MergeReport:
    """Fold `source`'s state into `target`.

    The target agent is mutated. The source is read-only.
    Returns a MergeReport describing what was added vs reinforced.
    """
    report = MergeReport()

    # ---- skills --------------------------------------------------------
    for sig, s_skill in source.skills.skills.items():
        existing = target.skills.skills.get(sig)
        if existing is None:
            # Copy by re-recording successes / failures.
            from rck.skills import Skill
            target.skills.skills[sig] = Skill(
                name=s_skill.name,
                pattern=list(s_skill.pattern),
                success_count=s_skill.success_count,
                failure_count=s_skill.failure_count,
                last_used=s_skill.last_used,
            )
            report.skills_added += 1
        else:
            existing.success_count += s_skill.success_count
            existing.failure_count += s_skill.failure_count
            existing.last_used = max(existing.last_used, s_skill.last_used)
            report.skills_reinforced += 1

    # ---- provenance ----------------------------------------------------
    for key, s_rec in source.provenance._records.items():
        existing = target.provenance._records.get(key)
        if existing is None:
            from copy import deepcopy
            target.provenance._records[key] = deepcopy(s_rec)
            report.provenance_added += 1
        else:
            existing.count += s_rec.count
            existing.last_seen = max(existing.last_seen, s_rec.last_seen)
            existing.tags.update(s_rec.tags)
            if (existing.source != s_rec.source
                    and existing.source != "multi"):
                existing.source = "multi"
            existing.confidence = min(
                1.0,
                existing.confidence + (1.0 - existing.confidence) * 0.1,
            )
            report.provenance_reinforced += 1

    # ---- knowledge base ------------------------------------------------
    # Per-shard bundle addition. Assumes same shard count and dim.
    if (target.knowledge.n_shards == source.knowledge.n_shards
            and target.knowledge.dim == source.knowledge.dim):
        for i, src_shard in enumerate(source.knowledge._shards):
            target.knowledge._shards[i].merge(src_shard)
        report.kb_facts_merged = source.knowledge.size()
        target.knowledge._fact_count += source.knowledge.size()

    # Cache: incoming facts may have created shorter chains.
    if target.chain_cache is not None:
        target.chain_cache.bump_version()

    return report
