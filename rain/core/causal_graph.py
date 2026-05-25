# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Causal graph reasoning over the KB.

Pearl-style causal inference layer on top of RAIN-Net's semantic memory.
Triples (subject, relation, object) are extracted from KB facts via
light parsing; the resulting graph supports:

    - do(X)=Y interventions: "if X were Y, what would Z be?"
    - ancestor/descendant queries: "what causes X?", "what does X cause?"
    - cycle detection
    - shortest causal chain between two nodes

This is a real capability LLMs and standard RAG do not have. They can
retrieve facts about causation but cannot reason about counterfactual
interventions in a principled way.

Why HV-substrate-aware: each node has an HV identity, so the graph
operations compose with the rest of RAIN-Net. An intervention "do(X=Y)"
binds the new value into the node HV; downstream nodes are recomputed
via message passing on the graph.

Implementation is intentionally lightweight (pure Python + networkx-like
graph). Production-scale would back this with neo4j or similar.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hv_substrate import (
    DEFAULT_DIM,
    bind,
    bundle,
    hash_to_hv,
    similarity,
)


# Light SVO regex for triple extraction from natural-language fact text.
_SVO_RX = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[a-z]+)*?)\s+"
    r"(causes?|leads? to|results? in|requires?|implies?|prevents?|"
    r"increases?|decreases?|is associated with|is caused by|"
    r"depends? on|affects?|is a|are a)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[a-z]+)*?)\b"
)


@dataclass
class CausalEdge:
    """A directed edge in the causal graph."""

    source: str
    target: str
    relation: str
    fact_id: str = ""
    confidence: float = 1.0


@dataclass
class CausalNode:
    name: str
    hv: np.ndarray
    incoming: list[CausalEdge] = field(default_factory=list)
    outgoing: list[CausalEdge] = field(default_factory=list)


@dataclass
class CausalGraph:
    """A directed graph of causal-relation triples extracted from a KB.

    Build via add_triples_from_facts(); query via ancestors/descendants/
    do_intervention/causal_chain.
    """

    dim: int = DEFAULT_DIM
    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)

    # ---------- building ----------

    def add_node(self, name: str) -> CausalNode:
        """Idempotent. Returns the node."""
        key = name.strip().lower()
        if key in self.nodes:
            return self.nodes[key]
        node = CausalNode(name=name.strip(), hv=hash_to_hv(f"causal::{key}", dim=self.dim))
        self.nodes[key] = node
        return node

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        fact_id: str = "",
        confidence: float = 1.0,
    ) -> CausalEdge:
        s_node = self.add_node(source)
        t_node = self.add_node(target)
        edge = CausalEdge(
            source=s_node.name,
            target=t_node.name,
            relation=relation,
            fact_id=fact_id,
            confidence=confidence,
        )
        s_node.outgoing.append(edge)
        t_node.incoming.append(edge)
        self.edges.append(edge)
        return edge

    def add_triples_from_text(self, text: str, fact_id: str = "") -> int:
        """Extract SVO triples from a single fact text and add as edges.

        Returns the number of edges added.
        """
        n = 0
        for m in _SVO_RX.finditer(text):
            s, r, o = m.group(1), m.group(2), m.group(3)
            # Skip degenerate "X is a Y" with too-generic class for now.
            if r.strip().lower() in {"is a", "are a"} and len(o.split()) > 3:
                continue
            self.add_edge(s, r, o, fact_id=fact_id)
            n += 1
        return n

    def add_triples_from_kb(self, kb) -> int:
        """Bulk-add: iterate semantic memory and extract triples per fact.

        kb is a rain.core.hierarchical_memory.SemanticMemory instance.
        """
        total = 0
        for fid, fact in kb._facts.items():
            total += self.add_triples_from_text(fact.text, fact_id=fid)
        return total

    # ---------- queries ----------

    def ancestors(self, name: str, max_depth: int = 5) -> list[tuple[str, int]]:
        """Return (ancestor_name, depth) for all ancestors of name.

        Useful for "what causes X" style queries.
        """
        key = name.strip().lower()
        if key not in self.nodes:
            return []
        visited: dict[str, int] = {}
        q: deque[tuple[str, int]] = deque([(key, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_depth:
                continue
            for edge in self.nodes[cur].incoming:
                src_key = edge.source.lower()
                if src_key in visited and visited[src_key] <= d + 1:
                    continue
                visited[src_key] = d + 1
                q.append((src_key, d + 1))
        out = [(self.nodes[k].name, depth) for k, depth in visited.items()]
        out.sort(key=lambda x: x[1])
        return out

    def descendants(self, name: str, max_depth: int = 5) -> list[tuple[str, int]]:
        """Return (descendant_name, depth) for all descendants of name.

        Useful for "what does X cause / lead to" style queries.
        """
        key = name.strip().lower()
        if key not in self.nodes:
            return []
        visited: dict[str, int] = {}
        q: deque[tuple[str, int]] = deque([(key, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_depth:
                continue
            for edge in self.nodes[cur].outgoing:
                tgt_key = edge.target.lower()
                if tgt_key in visited and visited[tgt_key] <= d + 1:
                    continue
                visited[tgt_key] = d + 1
                q.append((tgt_key, d + 1))
        out = [(self.nodes[k].name, depth) for k, depth in visited.items()]
        out.sort(key=lambda x: x[1])
        return out

    def causal_chain(self, source: str, target: str) -> list[CausalEdge] | None:
        """Find the shortest causal chain from source to target.

        Returns the list of edges in path order, or None if no path.
        """
        s_key = source.strip().lower()
        t_key = target.strip().lower()
        if s_key not in self.nodes or t_key not in self.nodes:
            return None
        # BFS shortest path on the directed graph.
        prev: dict[str, CausalEdge] = {}
        q: deque[str] = deque([s_key])
        visited: set[str] = {s_key}
        while q:
            cur = q.popleft()
            if cur == t_key:
                # Reconstruct path.
                chain: list[CausalEdge] = []
                node = t_key
                while node in prev:
                    edge = prev[node]
                    chain.append(edge)
                    node = edge.source.lower()
                chain.reverse()
                return chain
            for edge in self.nodes[cur].outgoing:
                tgt = edge.target.lower()
                if tgt not in visited:
                    visited.add(tgt)
                    prev[tgt] = edge
                    q.append(tgt)
        return None

    def do_intervention(
        self,
        node_name: str,
        new_value_hv: np.ndarray,
        max_propagation_depth: int = 3,
    ) -> dict[str, np.ndarray]:
        """Pearl-style do(X)=value: replace node X's HV with new_value_hv,
        then propagate through descendants for up to max_propagation_depth
        levels.

        Returns a dict of {node_name: new_hv} after propagation.

        This is intentionally simple (additive bundle propagation); full
        Pearl do-calculus would require SCM with deterministic functions.
        v0.2 demonstrates the substrate-aware causal layer; v0.3 will
        add structural causal models for proper interventional inference.
        """
        key = node_name.strip().lower()
        if key not in self.nodes:
            return {}
        # Set the intervened node.
        result: dict[str, np.ndarray] = {self.nodes[key].name: new_value_hv.astype(np.float32)}
        # BFS-propagate through descendants.
        q: deque[tuple[str, int]] = deque([(key, 0)])
        visited: set[str] = {key}
        while q:
            cur_key, depth = q.popleft()
            if depth >= max_propagation_depth:
                continue
            cur_hv = result[self.nodes[cur_key].name]
            for edge in self.nodes[cur_key].outgoing:
                tgt_key = edge.target.lower()
                if tgt_key in visited:
                    continue
                visited.add(tgt_key)
                # Propagated HV: bind the cause HV with the relation HV.
                rel_hv = hash_to_hv(f"relation::{edge.relation}", dim=self.dim)
                propagated = bind(cur_hv, rel_hv)
                result[self.nodes[tgt_key].name] = propagated
                q.append((tgt_key, depth + 1))
        return result

    # ---------- diagnostics ----------

    def has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(lambda: WHITE)

        def visit(n: str) -> bool:
            if color[n] == GRAY:
                return True  # back edge
            if color[n] == BLACK:
                return False
            color[n] = GRAY
            for edge in self.nodes[n].outgoing:
                if visit(edge.target.lower()):
                    return True
            color[n] = BLACK
            return False

        for k in self.nodes:
            if color[k] == WHITE and visit(k):
                return True
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "has_cycle": self.has_cycle(),
            "relation_counts": dict(
                __import__("collections").Counter(e.relation for e in self.edges)
            ),
        }
