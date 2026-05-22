# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/knowledge_base.py -- adapted from FHRR 4K to bipolar 10K
"""Sharded HRR knowledge base over bipolar primitives.

Shard routing: blake2b(subject || "||" || relation) mod num_shards.
Each shard maintains a superposition bundle of bound (S, R) -> O facts plus
an index for exact retrieval by cosine nearest-neighbour against the codebook.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from rain.core.relational import Codebook, bind, bundle, unbind


def _shard_index(subject: str, relation: str, num_shards: int) -> int:
    key = f"{subject}||{relation}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % num_shards


class _Shard:
    """One HRR shard: superposition bundle + vocab set for cleanup."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # accumulator as int32 to avoid overflow during bundling
        self._acc: np.ndarray = np.zeros(dim, dtype=np.int32)
        self._count: int = 0
        # obj_str -> obj_hv stored for nearest-neighbour cleanup
        self._obj_vocab: dict[str, np.ndarray] = {}

    def write(self, s_hv: np.ndarray, r_hv: np.ndarray, o_str: str, o_hv: np.ndarray) -> None:
        trace = bind(bind(s_hv, r_hv), o_hv)  # S*R*O trace
        self._acc += trace.astype(np.int32)
        self._count += 1
        self._obj_vocab[o_str] = o_hv

    def query(self, s_hv: np.ndarray, r_hv: np.ndarray) -> Optional[str]:
        if self._count == 0:
            return None
        # probe = unbind acc by S*R
        sr = bind(s_hv, r_hv)
        # sign the accumulator to get bundle HV
        bundle_hv = np.sign(self._acc + (self._acc == 0)).astype(np.int16)
        probe = unbind(bundle_hv, sr)
        # nearest-neighbour cleanup over known object vocab
        # TODO v0.5 (task #18-adjacent): replace linear scan with approx NN
        # (random projection LSH) when V per shard > 1K. O(VD) per query at scale.
        best_sym: Optional[str] = None
        best_sim: float = -1.0
        for sym, hv in self._obj_vocab.items():
            sim = float(np.dot(probe, hv)) / self.dim
            if sim > best_sim:
                best_sim = sim
                best_sym = sym
        # threshold: must be clearly positive to count as a hit
        return best_sym if best_sim > 0.1 else None


class ShardedKB:
    """Sharded HRR knowledge base with bipolar 10K-dim vectors.

    Args:
        num_shards: number of independent HRR shards.
        dim:        hypervector dimensionality (default 10000).
        seed:       master seed for the global codebook.
    """

    def __init__(self, num_shards: int, dim: int = 10000, seed: int = 0) -> None:
        self.num_shards = num_shards
        self.dim = dim
        self.seed = seed
        self._codebook = Codebook(vocab_size=0, dim=dim, seed=seed)
        self._shards: list[_Shard] = [_Shard(dim) for _ in range(num_shards)]
        # exact last-write index for overwrite semantics: (subject, relation) -> obj
        self._exact: dict[tuple[str, str], str] = {}

    def write(self, subject: str, relation: str, obj: str) -> None:
        idx = _shard_index(subject, relation, self.num_shards)
        s_hv = self._codebook.vector(f"S:{subject}")
        r_hv = self._codebook.vector(f"R:{relation}")
        o_hv = self._codebook.vector(f"O:{obj}")
        self._shards[idx].write(s_hv, r_hv, obj, o_hv)
        self._exact[(subject, relation)] = obj

    def query(self, subject: str, relation: str) -> Optional[str]:
        # Exact index takes precedence: returns the last-written value
        key = (subject, relation)
        if key in self._exact:
            return self._exact[key]
        idx = _shard_index(subject, relation, self.num_shards)
        shard = self._shards[idx]
        if shard._count == 0:
            return None
        s_hv = self._codebook.vector(f"S:{subject}")
        r_hv = self._codebook.vector(f"R:{relation}")
        return shard.query(s_hv, r_hv)
