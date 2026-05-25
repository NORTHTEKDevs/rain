# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HV substrate: the canonical Vector-Symbolic-Architecture primitives
that every RAIN-Net module imports.

Every other module in RAIN-Net (encoder bank, MoA router, hierarchical
memory, verifier head, symbolic verifier, KB-Attention) speaks in
hypervectors produced by these ops. The substrate guarantees that an
HV produced by the image encoder lives in the same space as one
produced by the text encoder, which lives in the same space as a
fact HV in semantic memory, which lives in the same space as the
domain HV of the Tsetlin expert.

This unification is what makes RAIN-Net a coherent architecture instead
of a pile of techniques. Without a common substrate, each module would
need a translation layer to every other module -- O(N^2) integration
cost. With this substrate, every new module costs O(1) to plug in.

Primitive ops (Plate 1995 / Kanerva 2009):
    bind     a * b           elementwise product; its own inverse
    bundle   sign(a + b)     similarity-preserving superposition
    permute  roll(a, k)      fixed circular shift; position role
    cleanup  argmax cos(a,C) project to nearest codebook entry
    similar  cos(a, b)       semantic similarity in [-1, +1]

Convention: bipolar {-1, +1}^D at inference; tanh-relaxed during
training to keep gradients flowing.

D = 10000 is the default. Capacity ~D/2 ln(D) = ~46000 distinguishable
items at D=10K. For LLM-scale knowledge we use hierarchical HV banks
(see rain.core.hierarchical_memory for the partition strategy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DIM = 10_000


# -------------------- core primitives --------------------


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind: elementwise product. a (X) b. Its own inverse: (a*b)*b = a."""
    return (a * b).astype(np.float32)


def unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unbind: same as bind for bipolar HVs (since b*b = 1)."""
    return bind(c, b)


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """Bundle: similarity-preserving sum. Returns sign of sum.

    sign(0) is ambiguous; we resolve to +1 to keep bipolar property.
    """
    if not vectors:
        raise ValueError("bundle() needs at least one vector")
    s = np.sum(vectors, axis=0).astype(np.float32)
    out = np.sign(s)
    out = np.where(out == 0.0, 1.0, out).astype(np.float32)
    return out


def bundle_weighted(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted bundle (for soft memory readout, MoA aggregation)."""
    if len(vectors) != len(weights):
        raise ValueError("len(vectors) must equal len(weights)")
    s = np.zeros_like(vectors[0], dtype=np.float32)
    for v, w in zip(vectors, weights, strict=True):
        s += float(w) * v
    out = np.sign(s)
    out = np.where(out == 0.0, 1.0, out).astype(np.float32)
    return out


def permute(a: np.ndarray, k: int = 1) -> np.ndarray:
    """Permute: fixed circular shift by k. Acts as a free position role.

    permute(a, k1) and permute(a, k2) are near-orthogonal for k1 != k2
    when D is large, so positions are distinguishable without learned
    parameters.
    """
    return np.roll(a, k).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in HV space. Returns float in [-1, +1]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Batched cosine similarity. queries (M, D), keys (N, D) -> (M, N)."""
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    kn = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9)
    return (qn @ kn.T).astype(np.float32)


# -------------------- random / seeded HVs --------------------


def random_hv(dim: int = DEFAULT_DIM, seed: int | None = None) -> np.ndarray:
    """Sample a single random bipolar HV. Used for fresh role vectors."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=dim) * 2 - 1).astype(np.float32)


def random_hv_batch(n: int, dim: int = DEFAULT_DIM, seed: int | None = None) -> np.ndarray:
    """Sample n random bipolar HVs of shape (n, dim)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(n, dim)) * 2 - 1).astype(np.float32)


def hash_to_hv(key: str | bytes, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic HV from a string/bytes key. Reproducible across runs.

    Used to give role HVs that survive process restarts -- e.g. the role
    HV for "position 5" must be the same vector tomorrow as it was
    today. Standard random_hv() would change with a different RNG seed.
    """
    import hashlib

    if isinstance(key, str):
        key = key.encode("utf-8")
    h = hashlib.sha256(key).digest()
    seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return random_hv(dim, seed=seed ^ dim)


# -------------------- codebook + cleanup --------------------


@dataclass
class Codebook:
    """A bank of named HVs supporting cleanup (nearest-neighbour match).

    The codebook is the "vocabulary" of distinguishable concepts the
    substrate knows about. Every symbol (token, fact id, expert name,
    modality role) gets a row.
    """

    dim: int
    seed: int = 0
    _items: dict[str, int] | None = None
    _vectors: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._items = {}
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, name: str) -> np.ndarray:
        """Add a name to the codebook (or return existing HV)."""
        assert self._items is not None and self._vectors is not None
        if name in self._items:
            return self._vectors[self._items[name]]
        hv = hash_to_hv(f"{self.seed}::{name}", dim=self.dim)
        idx = len(self._items)
        self._items[name] = idx
        self._vectors = np.vstack([self._vectors, hv[np.newaxis, :]])
        return hv

    def get(self, name: str) -> np.ndarray:
        """Look up an HV (raises if missing)."""
        assert self._items is not None and self._vectors is not None
        if name not in self._items:
            raise KeyError(f"codebook miss: {name!r}")
        return self._vectors[self._items[name]]

    def cleanup(self, noisy: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """Find the top-k codebook entries closest to noisy HV."""
        assert self._items is not None and self._vectors is not None
        if len(self._items) == 0:
            return []
        sims = similarity_matrix(noisy[np.newaxis, :], self._vectors)[0]
        idx = np.argsort(-sims)[:top_k]
        names = list(self._items.keys())
        return [(names[i], float(sims[i])) for i in idx]

    def __len__(self) -> int:
        assert self._items is not None
        return len(self._items)


# -------------------- composite ops (built from primitives) --------------------


def role_filler(role_hv: np.ndarray, filler_hv: np.ndarray) -> np.ndarray:
    """Encode a role-filler pair: role (X) filler. Standard VSA pattern.

    Example: bind(role_subject, filler_apollo) = "subject is apollo".
    Multiple role-fillers bundled = a complete relational fact.
    """
    return bind(role_hv, filler_hv)


def encode_triple(s: np.ndarray, r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Encode a (subject, relation, object) triple as a single HV.

    bundle(bind(role_subject, s), bind(role_relation, r), bind(role_object, o))
    """
    role_s = hash_to_hv("role::subject", dim=s.shape[0])
    role_r = hash_to_hv("role::relation", dim=s.shape[0])
    role_o = hash_to_hv("role::object", dim=s.shape[0])
    return bundle(bind(role_s, s), bind(role_r, r), bind(role_o, o))


def encode_sequence(items: list[np.ndarray]) -> np.ndarray:
    """Encode an ordered sequence using permute as position role.

    seq_hv = bundle(items[0], permute(items[1], 1), permute(items[2], 2), ...)
    """
    if not items:
        raise ValueError("empty sequence")
    return bundle(*[permute(v, k) for k, v in enumerate(items)])


def encode_set(items: list[np.ndarray]) -> np.ndarray:
    """Encode an unordered set: just bundle. Order-independent by design."""
    if not items:
        raise ValueError("empty set")
    return bundle(*items)


def encode_kv_map(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Encode a key-value map as bundle of bound pairs. Retrieval is unbind."""
    return bundle(*[bind(k, v) for k, v in pairs])


def query_kv(map_hv: np.ndarray, key_hv: np.ndarray) -> np.ndarray:
    """Query a KV-map encoded HV: unbind(map, key) ~ noisy value."""
    return unbind(map_hv, key_hv)


# -------------------- training-friendly soft variants --------------------


def soft_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Soft (differentiable) bind: same op, but caller passes float HVs.

    Used inside torch graphs where we relax bipolar to tanh outputs.
    Numerically identical to bind() at float precision.
    """
    return (a * b).astype(np.float32)


def soft_bundle(*vectors: np.ndarray) -> np.ndarray:
    """Soft bundle: just sum, no sign(). Differentiable.

    Discretise to bipolar at evaluation via bundle(*).
    """
    return np.sum(vectors, axis=0).astype(np.float32)


def bipolarize(x: np.ndarray) -> np.ndarray:
    """Sign-quantise to bipolar {-1, +1}. Inference-time finalisation."""
    out = np.sign(x).astype(np.float32)
    out = np.where(out == 0.0, 1.0, out).astype(np.float32)
    return out
