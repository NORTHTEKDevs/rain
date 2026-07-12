"""Theory of Mind: per-agent belief stores separate from ground truth.

Each external agent (Alice, Bob, ...) has their own sharded HRR KB. Believes
of agent X about (S, R) -> O are stored as belief.write(X, S, R, O). Beliefs
about an agent's beliefs (second-order ToM) are stored at depth = 2.

This implementation supports first-order ToM (X believes that S R O) and
second-order ToM (X believes that Y believes that S R O) via nested-KB
addressing.
"""

from __future__ import annotations

import hashlib

from rain.core.knowledge_base import ShardedKB


class TheoryOfMind:
    def __init__(self, dim: int = 10000, num_shards: int = 8, seed: int = 0) -> None:
        self.dim = dim
        self.num_shards = num_shards
        self.seed = seed
        # One KB per (believer-chain): key is tuple of strings, value is ShardedKB
        self._kbs: dict[tuple[str, ...], ShardedKB] = {}

    def _kb_for(self, believers: tuple[str, ...]) -> ShardedKB:
        if believers not in self._kbs:
            # Derive a chain seed deterministically across Python processes via blake2b.
            # python hash() is NOT deterministic across processes (PYTHONHASHSEED varies).
            h = hashlib.blake2b(
                ("|".join(believers) + f":{self.seed}").encode(), digest_size=8
            ).digest()
            chain_seed = int.from_bytes(h, "big") & 0xFFFFFFFF
            self._kbs[believers] = ShardedKB(
                num_shards=self.num_shards, dim=self.dim, seed=chain_seed
            )
        return self._kbs[believers]

    def believe(self, believer: str, subject: str, relation: str, object_: str) -> None:
        """First-order: believer believes that (subject, relation) -> object_."""
        self._kb_for((believer,)).write(subject, relation, object_)

    def query_belief(self, believer: str, subject: str, relation: str) -> str | None:
        return self._kb_for((believer,)).query(subject, relation)

    def believe_second_order(
        self, outer: str, inner: str, subject: str, relation: str, object_: str
    ) -> None:
        """outer believes that inner believes that (S, R) -> O."""
        self._kb_for((outer, inner)).write(subject, relation, object_)

    def query_second_order(self, outer: str, inner: str, subject: str, relation: str) -> str | None:
        return self._kb_for((outer, inner)).query(subject, relation)

    def believers(self) -> list[str]:
        """List of first-order believers."""
        return sorted({chain[0] for chain in self._kbs if len(chain) == 1})
