"""Episodic-to-procedural consolidation ("dreaming").

Background pass that walks the QueryMemory log and turns repeated,
stable query patterns into proactive cache entries and skill
records. The biological analogue is hippocampal replay during sleep
-- frequently-seen episodes get consolidated into longer-term,
faster-to-access form.

What this does:
  1. Find query signatures that fired >= N times AND consistently
     produced the same top-symbol (no drift).
  2. For each stable signature, ensure the (start, target=top_sym)
     chain is in the chain_cache (warm path).
  3. If the signature's epistemic state was "ambiguous" the last
     few times, log it for the user to disambiguate (idk gap).

Compared to `warm_cache_from_history` (iter 31) which used HOT
signatures regardless of stability, this consolidator focuses on
STABLE signatures -- ones the agent has consistently answered the
same way -- and skips unstable ones.

Output: a `ConsolidationReport` summarising what was promoted vs
flagged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rck.chain_cache import ChainCache
from rck.knowledge_base import ShardedKnowledgeBase
from rck.query_memory import QueryEpisode, QueryMemory


@dataclass
class ConsolidationReport:
    stable_promoted: list[tuple[str, str]] = field(default_factory=list)
    unstable_flagged: list[tuple] = field(default_factory=list)
    ambiguous_flagged: list[tuple] = field(default_factory=list)
    n_signatures_examined: int = 0

    def total_promoted(self) -> int:
        return len(self.stable_promoted)


def consolidate(memory: QueryMemory,
                *,
                min_occurrences: int = 3,
                stability_threshold: float = 0.9,
                kb: ShardedKnowledgeBase | None = None,
                chain_cache: ChainCache | None = None,
                discover_fn=None,
                ) -> ConsolidationReport:
    """Walk the query memory, group by signature, and report which
    signatures are STABLE enough to consolidate.

    Args:
      min_occurrences: ignore signatures seen fewer times.
      stability_threshold: fraction of episodes for a signature that
        must agree on the same top_symbol. Below this, the signature
        is flagged as UNSTABLE rather than promoted.
      discover_fn: optional callable `(s, target) -> spec`. When
        provided AND chain_cache is provided, stable signatures
        trigger a chain discovery and the resulting spec is stored
        in the cache.
    """
    report = ConsolidationReport()
    # Group by (frozen-known, unknown_role).
    groups: dict[tuple, list[QueryEpisode]] = defaultdict(list)
    for e in memory.all():
        sig = (tuple(sorted(e.known.items())), e.unknown_role)
        groups[sig].append(e)
    report.n_signatures_examined = len(groups)

    for sig, episodes in groups.items():
        if len(episodes) < min_occurrences:
            continue
        # Count top_symbols among NON-idk episodes.
        non_idk = [e for e in episodes if e.state != "idk"]
        if not non_idk:
            continue
        sym_counts: dict = defaultdict(int)
        for e in non_idk:
            if e.top_symbol is not None:
                sym_counts[str(e.top_symbol).lower()] += 1
        if not sym_counts:
            continue
        total = sum(sym_counts.values())
        top_sym, top_count = max(sym_counts.items(), key=lambda kv: kv[1])
        stability = top_count / total
        known = dict(sig[0])
        s = str(known.get("S", "")).lower()
        # Stable, non-idk: promote.
        if stability >= stability_threshold and s:
            report.stable_promoted.append((s, top_sym))
            if (chain_cache is not None and kb is not None
                    and discover_fn is not None
                    and chain_cache.get(s, top_sym) is None):
                spec = discover_fn(s, top_sym)
                # discover_fn handles cache insertion itself.
                _ = spec
        elif stability < stability_threshold:
            report.unstable_flagged.append((s, sig[1], dict(sym_counts)))
        # Ambiguous tracking.
        ambiguous_share = sum(1 for e in episodes
                               if e.state == "ambiguous") / len(episodes)
        if ambiguous_share >= 0.5:
            report.ambiguous_flagged.append((s, sig[1], ambiguous_share))
    return report
