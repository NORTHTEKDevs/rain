"""Bridge between RAIN's ShardedKB (the persistent fact memory) and
HymnPlusV2's inline KB-Attention buffer (the model's per-block memory).

Why bridge them:
  * ShardedKB scales to 10K-1M+ facts; KB-Attention buffer is sized
    for the model (typically 1K-8K).
  * ShardedKB is the source of truth that tell() updates.
  * KB-Attention buffer is what the model sees at inference.

Two refresh strategies:
  1. Static: at session start, pick the top-N facts (by some heuristic)
     and load them into the KB-Attention buffer for the whole session.
  2. Dynamic: before each ask(), build a query hypervector and pull the
     top-N facts from ShardedKB that match it, then refresh the
     KB-Attention buffer with those N facts. The model then attends
     over the most relevant N facts per query.

This module ships strategy 1 (simple, fast) and the hook for strategy 2
(retrieval-conditioned KB refresh).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from rain.core.knowledge_base import ShardedKB
from rain.core.relational import Codebook, bind, bundle


def kb_to_hypervectors(
    kb: ShardedKB,
    triples: Iterable[tuple[str, str, str]],
    dim: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Encode (s, r, o) triples as (N, dim) bipolar hypervectors via the
    same bind+bundle path the v5 training uses. Reusable for both:

      * Building the KB-Attention buffer from ShardedKB content
      * Building a held-out validation KB for `validate_v2_kb_grounding`

    The Codebook seed must match across runs (training + inference) for
    fact hypervectors to be consistent. Defaults to 0 to match what
    `HymnPlusV2Sampler.set_kb_from_facts` uses.
    """
    cb = Codebook(vocab_size=16384, dim=dim, seed=seed)
    rows: list[np.ndarray] = []
    for s, r, o in triples:
        fact = bundle([bind(cb.vector(str(s)), cb.vector(str(r))), cb.vector(str(o))])
        rows.append(fact.astype(np.float32))
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows, axis=0)


def static_buffer_from_sharded_kb(
    kb: ShardedKB,
    target_size: int,
    dim: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Materialize up to `target_size` triples from ShardedKB into a
    (target_size, dim) bipolar buffer. Pads with random bipolar if KB
    has fewer triples.

    The ordering is whatever ShardedKB iterates in (insertion order in
    the v0 implementation). If we add a priority field later, sort by
    that here.
    """
    triples = list(kb.iter_triples()) if hasattr(kb, "iter_triples") else _harvest_triples(kb)
    triples = triples[:target_size]
    rows = list(kb_to_hypervectors(kb, triples, dim=dim, seed=seed))

    if len(rows) < target_size:
        rng = np.random.default_rng(seed + 1)
        pad_count = target_size - len(rows)
        pad = (rng.integers(0, 2, size=(pad_count, dim)) * 2 - 1).astype(np.float32)
        rows.extend(list(pad))

    return np.stack(rows[:target_size], axis=0)


def _harvest_triples(kb: ShardedKB) -> list[tuple[str, str, str]]:
    """Fallback: ShardedKB in current code doesn't expose triple iteration
    cleanly. We harvest by inspecting its shards directly. This is a
    best-effort path; the proper fix is to add ShardedKB.iter_triples().
    """
    out: list[tuple[str, str, str]] = []
    # ShardedKB tracks writes via a `_writes` log on each shard (v0 impl)
    # or via a separate _triples dict (depending on revision). Try both.
    for shard in getattr(kb, "shards", []):
        if hasattr(shard, "triples"):
            out.extend(shard.triples)
        elif hasattr(shard, "_triples"):
            out.extend(shard._triples)
    return out


class DynamicKbRefresher:
    """Strategy 2 hook: at query time, builds a query hypervector from
    (subject, relation) and refreshes the model's KB-Attention buffer
    with the top-N most-similar facts from ShardedKB.

    For now this is a stub that demonstrates the API. Real retrieval
    needs a fast nearest-neighbor index over ShardedKB content (FAISS,
    HNSW, or a sharded dot-product scan). Build this when KB grows
    beyond ~50K facts.
    """

    def __init__(self, kb: ShardedKB, model, *, top_n: int | None = None) -> None:
        self.kb = kb
        self.model = model
        # Default top_n = the model's KB-Attn buffer size
        first_kb_block = next((b for b in model.blocks if b.use_kb_attn), None)
        self.top_n = top_n or (first_kb_block.kb_attn.kb_size if first_kb_block else 1024)
        self.dim = first_kb_block.kb_attn.dim if first_kb_block else model.config.dim

    def refresh_for(self, subject: str, relation: str) -> int:
        """Pull the top_n most-relevant facts for (subject, relation),
        load them into the model's KB-Attention buffer, return how many
        facts were loaded.
        """
        import torch

        # v0: just take the first top_n facts. v1 will do proper
        # similarity search against a query hypervector built from
        # bind(subject, relation).
        triples = (
            list(self.kb.iter_triples())
            if hasattr(self.kb, "iter_triples")
            else _harvest_triples(self.kb)
        )
        triples = triples[: self.top_n]
        if not triples:
            return 0

        kb_np = kb_to_hypervectors(self.kb, triples, dim=self.dim)
        if kb_np.shape[0] < self.top_n:
            rng = np.random.default_rng(0)
            pad = (rng.integers(0, 2, size=(self.top_n - kb_np.shape[0], self.dim)) * 2 - 1).astype(
                np.float32
            )
            kb_np = np.concatenate([kb_np, pad], axis=0)

        self.model.set_kb(torch.as_tensor(kb_np[: self.top_n], dtype=torch.float32))
        return min(len(triples), self.top_n)
