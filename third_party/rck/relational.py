"""Relational memory built on VSA binding (Plate-style HRR).

A fact (slot_1=val_1, slot_2=val_2, ..., slot_k=val_k) is stored as the
ELEMENTWISE PRODUCT of role-bound filler hypervectors:

    fact_hv = prod_i [ bind(role_i, hv[val_i]) ]

The memory is the sum (bundle) of all such fact HVs.

To recover val_k given all other slots:

    key = prod_{i != k} [ bind(role_i, hv[val_i]) ]  * role_k
    estimate = memory * key
    answer = codebook.cleanup(estimate)

For the matching fact F the products cancel pairwise (bind is self-
inverse for bipolar HVs) and `F * key` reduces exactly to hv[val_k];
for non-matching facts F' the product is random noise. Summing gives
hv[val_k] plus crosstalk that the cleanup step rejects.

Capacity is ~ D / log(|codebook|) facts at 95% recall for D-dim bipolar
HVs. At D=4096 that comfortably covers a few hundred facts.

This is the substrate that lets RCK do something LLMs structurally
cannot: store and retrieve arbitrary structured knowledge without
fine-tuning, with O(1) insertion and exact compositional generalisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np

from rck.codebook import Codebook
from rck.vsa import bind, random_hv


@dataclass
class RelationalMemory:
    """Holographic Reduced Representation memory.

    Roles (E, R, V or arbitrary slot names) are themselves bipolar HVs.
    Facts are bundled additively; query is bind-then-cleanup.
    """

    dim: int = 4096
    seed: int = 0
    role_names: tuple[str, ...] = ("E", "R", "V")
    _roles: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _memory: np.ndarray = field(default=None, init=False)
    _facts: list[dict[str, Hashable]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Role HVs are derived from a name-keyed seed so that they CANNOT
        # collide with the codebook's atoms even when both share `seed`.
        for name in self.role_names:
            self._roles[name] = self._make_role(name)
        self._memory = np.zeros(self.dim, dtype=np.float32)

    def _make_role(self, name: str) -> np.ndarray:
        # PROCESS-STABLE salt via hashlib. Built-in hash() is randomised
        # between processes which would make role HVs change between runs.
        import hashlib
        key = f"rck-role-v1|{self.seed}|{name}".encode("utf-8")
        salt = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
        rng = np.random.default_rng(salt)
        return random_hv(self.dim, rng)

    # ---- core ops ----------------------------------------------------------

    def role(self, name: str) -> np.ndarray:
        if name not in self._roles:
            self._roles[name] = self._make_role(name)
        return self._roles[name]

    def _fact_hv(self, codebook: Codebook, fact: dict[str, Hashable]) -> np.ndarray:
        """Multiplicative binding of all (role, value) bindings in the fact."""
        encoded = np.ones(self.dim, dtype=np.int8)
        for role, symbol in fact.items():
            sym_hv = codebook.encode(symbol)
            bound = bind(self.role(role), sym_hv)
            encoded = bind(encoded, bound)
        return encoded

    def store(self, codebook: Codebook, fact: dict[str, Hashable]) -> None:
        """Insert a fact: dict of role -> symbol. Stored as bipolar product."""
        fact_hv = self._fact_hv(codebook, fact)
        self._memory += fact_hv.astype(np.float32)
        self._facts.append(dict(fact))

    def store_many(self, codebook: Codebook, facts: Iterable[dict[str, Hashable]]) -> None:
        for f in facts:
            self.store(codebook, f)

    def query(
        self,
        codebook: Codebook,
        known: dict[str, Hashable],
        unknown_role: str,
        top_k: int = 3,
    ) -> list[tuple[Hashable, float]]:
        """Given some role->symbol bindings, recover the value of the
        unknown role by unbinding the matching fact.

        key = prod over known: bind(role_i, codebook[sym_i])   * role[unknown]
        probe = memory * key
        cleanup(probe)
        """
        if unknown_role in known:
            raise ValueError("unknown_role cannot also be in known")
        key = np.ones(self.dim, dtype=np.int8)
        for role, sym in known.items():
            sym_hv = codebook.encode(sym)
            key = bind(key, bind(self.role(role), sym_hv))
        key = bind(key, self.role(unknown_role))
        probe = self._memory * key.astype(np.float32)
        return codebook.fast_cleanup(probe, top_k=top_k)

    def answer(
        self,
        codebook: Codebook,
        known: dict[str, Hashable],
        unknown_role: str,
    ) -> tuple[Hashable | None, float]:
        results = self.query(codebook, known, unknown_role, top_k=1)
        if not results:
            return None, 0.0
        return results[0]

    # ---- introspection -----------------------------------------------------

    def size(self) -> int:
        return len(self._facts)

    def facts(self) -> list[dict[str, Hashable]]:
        return list(self._facts)

    def merge(self, other: "RelationalMemory") -> None:
        """Bundle another memory's contents into this one.

        This is the federated-learning superpower: two memories trained on
        disjoint facts can be combined by adding their memory tensors.
        Role tags MUST match (same dim and same role HVs); we enforce
        equal dim and copy any extra roles from `other`.
        """
        if other.dim != self.dim:
            raise ValueError("dim mismatch")
        for name in other._roles:
            if name not in self._roles:
                self._roles[name] = other._roles[name].copy()
        self._memory = self._memory + other._memory
        self._facts.extend(other._facts)

    # ---- editable surgery --------------------------------------------------

    def forget(self, codebook: Codebook, fact: dict[str, Hashable]) -> None:
        """Subtract a fact from memory."""
        fact_hv = self._fact_hv(codebook, fact)
        self._memory -= fact_hv.astype(np.float32)
        for i, f in enumerate(self._facts):
            if f == fact:
                del self._facts[i]
                break

    def reset(self) -> None:
        self._memory = np.zeros(self.dim, dtype=np.float32)
        self._facts.clear()
