"""Theory of mind: belief tuples (believer, S, R, O).

A model passes a (very weak) theory-of-mind test if it can distinguish:
  - the world's ground truth: "Paris IS the capital of France"
  - a particular agent's belief:  "Bob THINKS the capital of France is Lyon"

Ground truth lives in the standard relational memory (3-tuple S/R/O).
Beliefs are stored with an extra "B" (believer) role, making them
4-tuples that don't pollute the ground-truth memory:

    store_belief("bob", "france", "capital", "lyon")
    -> {B: bob, S: france, R: capital, O: lyon}

Query patterns:
  - ground_truth(S, R)         -> O
  - what_does_x_think(B, S, R) -> O
  - everyone_who_believes(S, R, O) -> list of believers

This is the substrate for false-belief tasks (Sally-Anne, etc.) and
for any agent that needs to track distinct mental models of other
agents alongside its own world model.
"""
from __future__ import annotations

from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase


# We keep beliefs in their own KB so they cannot displace ground-truth
# facts during HRR cleanup.
def make_belief_kb(dim: int = 4096, n_shards: int = 64, seed: int = 0) -> ShardedKnowledgeBase:
    return ShardedKnowledgeBase(dim=dim, n_shards=n_shards, seed=seed)


def store_belief(kb: ShardedKnowledgeBase,
                 believer: Hashable, subject: Hashable,
                 relation: Hashable, value: Hashable) -> None:
    kb.store({"B": believer, "S": subject, "R": relation, "O": value})


def what_does_x_think(kb: ShardedKnowledgeBase,
                      believer: Hashable, subject: Hashable,
                      relation: Hashable, top_k: int = 3) -> list[tuple[Hashable, float]]:
    """Return believer's beliefs about (subject, relation)."""
    return kb.query(
        {"B": believer, "S": subject, "R": relation},
        "O",
        top_k=top_k,
    )


def believers_of(kb: ShardedKnowledgeBase,
                 subject: Hashable, relation: Hashable, value: Hashable,
                 top_k: int = 5) -> list[tuple[Hashable, float]]:
    """Who believes that (subject, relation) -> value?"""
    return kb.query(
        {"S": subject, "R": relation, "O": value},
        "B",
        top_k=top_k,
    )
