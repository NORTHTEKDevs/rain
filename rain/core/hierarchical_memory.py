# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Hierarchical memory: four addressable memory levels for RAIN-Net.

LLMs structurally have two kinds of memory: the activation state during
a forward pass (working memory) and the weights (frozen "long-term"
memory). That's it. Adding a fact means retraining.

RAIN-Net has four levels, all HV-addressable via cosine similarity:

    working      transient activations during the current forward pass
    episodic     content-addressable log of recent interactions
    semantic     KB facts as HVs (the existing ShardedKB)
    procedural   compiled skills (small adapter weights keyed by HV)

Every layer in the model can cross-attend into any of these levels via
KB-Attention. Reads are cheap (cosine sim against the bank). Writes are
free for episodic/semantic; procedural writes train a tiny adapter.

This is the architectural answer to "continual learning without
catastrophic forgetting": the base model never gets gradient updates
after pretraining. Everything new lives in the memory layers, which by
construction cannot forget anything they hold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hv_substrate import DEFAULT_DIM, similarity_matrix


# -------------------- working memory --------------------


@dataclass
class WorkingMemory:
    """Transient HV buffer for the current forward pass.

    Holds <=N intermediate HVs from the current query's processing.
    Cleared between queries. Useful for the scratchpad reasoner and
    multi-hop routing decisions.
    """

    capacity: int = 32
    dim: int = DEFAULT_DIM
    _buffer: deque[tuple[str, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.capacity)

    def write(self, tag: str, hv: np.ndarray) -> None:
        self._buffer.append((tag, hv.astype(np.float32)))

    def read(self, query_hv: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if not self._buffer:
            return []
        tags = [t for t, _ in self._buffer]
        bank = np.stack([h for _, h in self._buffer], axis=0)
        sims = similarity_matrix(query_hv[np.newaxis, :], bank)[0]
        idx = np.argsort(-sims)[:top_k]
        return [(tags[i], float(sims[i])) for i in idx]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# -------------------- episodic memory --------------------


@dataclass
class EpisodicEntry:
    timestamp: float
    query_hv: np.ndarray
    answer_hv: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicMemory:
    """Content-addressable log of recent (query, answer) pairs.

    Used for "what did we say 50 turns ago" style queries -- the kind a
    transformer's fixed context window cannot serve. Lookups are HV
    cosine, so semantically related episodes are retrieved even if
    surface form differs.

    Capacity is bounded; oldest entries are evicted FIFO. For unbounded
    history, promote candidates to semantic memory periodically.
    """

    capacity: int = 1024
    dim: int = DEFAULT_DIM
    _entries: deque[EpisodicEntry] = field(init=False)

    def __post_init__(self) -> None:
        self._entries = deque(maxlen=self.capacity)

    def record(
        self,
        query_hv: np.ndarray,
        answer_hv: np.ndarray,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        import time as _time

        ts = _time.time() if timestamp is None else timestamp
        self._entries.append(
            EpisodicEntry(
                timestamp=ts,
                query_hv=query_hv.astype(np.float32),
                answer_hv=answer_hv.astype(np.float32),
                metadata=metadata or {},
            )
        )

    def search(
        self, query_hv: np.ndarray, top_k: int = 3
    ) -> list[tuple[float, EpisodicEntry]]:
        if not self._entries:
            return []
        bank = np.stack([e.query_hv for e in self._entries], axis=0)
        sims = similarity_matrix(query_hv[np.newaxis, :], bank)[0]
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._entries[i]) for i in idx]

    def __len__(self) -> int:
        return len(self._entries)


# -------------------- semantic memory --------------------


@dataclass
class SemanticFact:
    fact_id: str
    hv: np.ndarray
    text: str  # human-readable form for citation
    source: str = ""
    confidence: float = 1.0
    timestamp: float = 0.0
    disputed: bool = False


@dataclass
class SemanticMemory:
    """KB facts as HVs. Grows continuously without bound (in practice
    bounded by HV capacity ~D/2 ln(D); see hv_substrate doc).

    Each fact stores: the HV (for retrieval), the text (for citation),
    a source (for provenance), confidence and dispute flags (for active
    learning). The fact_id is a stable string so external systems can
    reference facts across runs.

    Bulk add from existing ShardedKB via from_sharded_kb(); this is the
    bridge from the existing KB system to the new memory layer.
    """

    dim: int = DEFAULT_DIM
    _facts: dict[str, SemanticFact] = field(default_factory=dict)
    _bank: np.ndarray = field(init=False)
    _ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._bank = np.zeros((0, self.dim), dtype=np.float32)

    def add(
        self,
        fact_id: str,
        hv: np.ndarray,
        text: str,
        source: str = "",
        confidence: float = 1.0,
    ) -> None:
        import time as _time

        if fact_id in self._facts:
            # Update in place (idempotent add).
            self._facts[fact_id].hv = hv.astype(np.float32)
            self._facts[fact_id].text = text
            self._facts[fact_id].source = source
            self._facts[fact_id].confidence = confidence
            self._facts[fact_id].timestamp = _time.time()
            idx = self._ids.index(fact_id)
            self._bank[idx] = hv.astype(np.float32)
            return
        fact = SemanticFact(
            fact_id=fact_id,
            hv=hv.astype(np.float32),
            text=text,
            source=source,
            confidence=confidence,
            timestamp=_time.time(),
        )
        self._facts[fact_id] = fact
        self._ids.append(fact_id)
        self._bank = np.vstack([self._bank, hv[np.newaxis, :].astype(np.float32)])

    def search(
        self,
        query_hv: np.ndarray,
        top_k: int = 5,
        include_disputed: bool = False,
    ) -> list[tuple[float, SemanticFact]]:
        if not self._facts:
            return []
        sims = similarity_matrix(query_hv[np.newaxis, :], self._bank)[0]
        order = np.argsort(-sims)
        out: list[tuple[float, SemanticFact]] = []
        for i in order:
            fact = self._facts[self._ids[i]]
            if fact.disputed and not include_disputed:
                continue
            out.append((float(sims[i]), fact))
            if len(out) >= top_k:
                break
        return out

    def dispute(self, fact_id: str) -> None:
        if fact_id in self._facts:
            self._facts[fact_id].disputed = True

    def __len__(self) -> int:
        return len(self._facts)


# -------------------- procedural memory --------------------


@dataclass
class ProceduralSkill:
    skill_id: str
    domain_hv: np.ndarray  # query HV pattern that triggers this skill
    adapter_path: str  # path to small adapter weights on disk
    description: str = ""
    invocations: int = 0


@dataclass
class ProceduralMemory:
    """Compiled skills as small adapter weights keyed by HV.

    When the router fails to find a confident expert, the procedural
    memory is checked: is there a skill whose domain HV is close to the
    query HV? If so, load its adapter and run.

    Adapters are stored on disk (typically <10MB each LoRA-style); the
    in-memory ProceduralSkill just holds the routing HV + the path. We
    do NOT lazy-load adapters in this v0.1 stub -- caller is responsible
    for loading the path when needed.
    """

    dim: int = DEFAULT_DIM
    _skills: dict[str, ProceduralSkill] = field(default_factory=dict)
    _bank: np.ndarray = field(init=False)
    _ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._bank = np.zeros((0, self.dim), dtype=np.float32)

    def add(
        self,
        skill_id: str,
        domain_hv: np.ndarray,
        adapter_path: str,
        description: str = "",
    ) -> None:
        skill = ProceduralSkill(
            skill_id=skill_id,
            domain_hv=domain_hv.astype(np.float32),
            adapter_path=adapter_path,
            description=description,
        )
        self._skills[skill_id] = skill
        self._ids.append(skill_id)
        self._bank = np.vstack(
            [self._bank, domain_hv[np.newaxis, :].astype(np.float32)]
        )

    def search(
        self, query_hv: np.ndarray, top_k: int = 3, threshold: float = 0.3
    ) -> list[tuple[float, ProceduralSkill]]:
        """Find skills whose domain HV matches the query.

        Threshold gates: only return skills above the threshold (so we
        don't fire skills that aren't really applicable).
        """
        if not self._skills:
            return []
        sims = similarity_matrix(query_hv[np.newaxis, :], self._bank)[0]
        order = np.argsort(-sims)
        out: list[tuple[float, ProceduralSkill]] = []
        for i in order:
            if sims[i] < threshold:
                break
            out.append((float(sims[i]), self._skills[self._ids[i]]))
            if len(out) >= top_k:
                break
        return out

    def record_invocation(self, skill_id: str) -> None:
        if skill_id in self._skills:
            self._skills[skill_id].invocations += 1

    def __len__(self) -> int:
        return len(self._skills)


# -------------------- unified facade --------------------


@dataclass
class HierarchicalMemory:
    """All four levels under one facade. The thing RAIN-Net.forward()
    actually talks to.

    Standard query pattern:
        results = memory.query_all(query_hv, top_k=3)
        # returns {'working': [...], 'episodic': [...],
        #          'semantic': [...], 'procedural': [...]}
    """

    dim: int = DEFAULT_DIM
    working: WorkingMemory = field(init=False)
    episodic: EpisodicMemory = field(init=False)
    semantic: SemanticMemory = field(init=False)
    procedural: ProceduralMemory = field(init=False)

    def __post_init__(self) -> None:
        self.working = WorkingMemory(dim=self.dim)
        self.episodic = EpisodicMemory(dim=self.dim)
        self.semantic = SemanticMemory(dim=self.dim)
        self.procedural = ProceduralMemory(dim=self.dim)

    def query_all(
        self, query_hv: np.ndarray, top_k: int = 3
    ) -> dict[str, list[tuple[float, Any]]]:
        return {
            "working": [(s, t) for t, s in self.working.read(query_hv, top_k=top_k)],
            "episodic": self.episodic.search(query_hv, top_k=top_k),
            "semantic": self.semantic.search(query_hv, top_k=top_k),
            "procedural": self.procedural.search(query_hv, top_k=top_k),
        }

    def stats(self) -> dict[str, int]:
        return {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "procedural": len(self.procedural),
        }
