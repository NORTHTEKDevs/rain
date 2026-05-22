# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/sofar/mapper.py -- transplanted from transformer
# residual-stream matrices to VSA codebook + role + HYMN-input matrices.
"""SVD mapper for RAIN routing.

Operates on the stacked rows of [codebook, role_matrix, W_hymn_in] (the
matrices we read FROM and write TO the VSA state) to extract top-k principal
routing directions in D-dim hypervector space.

Cached. Recomputation triggered when the underlying matrices change by
more than `rebuild_threshold` fraction of rows.
"""

from __future__ import annotations
import hashlib
import numpy as np
from dataclasses import dataclass


@dataclass
class RoutingDirections:
    """Result of SVD on stacked read+write matrices.

    Attributes:
        U: left singular vectors, shape (n_rows, k)
        sigma: singular values, shape (k,)
        V_T: right singular vectors (the routing directions in D-dim space),
             shape (k, D)
        k: number of directions kept
        D: hypervector dimension
        source_hash: blake2b digest of the stacked source matrices
    """
    U: np.ndarray
    sigma: np.ndarray
    V_T: np.ndarray
    k: int
    D: int
    source_hash: bytes


def _hash_source(stacked: np.ndarray) -> bytes:
    """Lightweight fingerprint for cache invalidation. NOT cryptographic.
    Detects shape changes + content changes through corner samples + sum."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(stacked.shape).encode())
    # Sample 16 corners/edges
    flat = stacked.flatten()
    n = flat.shape[0]
    if n >= 16:
        samples = flat[np.linspace(0, n - 1, 16, dtype=int)]
    else:
        samples = flat
    h.update(samples.tobytes())
    # Plus aggregate sum (catches changes that miss the samples)
    h.update(np.array([stacked.sum(), stacked.std()], dtype=np.float32).tobytes())
    return h.digest()


def compute_routing_directions(
    codebook_matrix: np.ndarray,
    role_matrix: np.ndarray | None,
    W_hymn_in: np.ndarray | None,
    k: int = 64,
) -> RoutingDirections:
    """SVD over stacked rows. Returns top-k routing directions.

    Args:
        codebook_matrix: (vocab_size, D) -- codebook vectors (bipolar int16
            or any numeric; cast to float32 for SVD).
        role_matrix: (num_roles, D) optional -- role hypervectors.
        W_hymn_in: (D, hidden_dim) optional -- HYMN input weight matrix. Its
            ROWS (each of length hidden_dim) are NOT in D-dim. Instead we
            transpose so each row IS a length-D vector: `W_hymn_in.T` of
            shape (hidden_dim, D).
        k: number of singular directions to keep.

    Returns:
        RoutingDirections with V_T shape (k, D) -- the principal routing axes.
    """
    rows: list[np.ndarray] = [codebook_matrix.astype(np.float32)]
    D = codebook_matrix.shape[1]
    if role_matrix is not None:
        if role_matrix.shape[1] != D:
            raise ValueError(f"role_matrix dim {role_matrix.shape[1]} != codebook D {D}")
        rows.append(role_matrix.astype(np.float32))
    if W_hymn_in is not None:
        if W_hymn_in.shape[0] != D:
            raise ValueError(f"W_hymn_in.shape[0]={W_hymn_in.shape[0]} != codebook D {D}")
        rows.append(W_hymn_in.T.astype(np.float32))

    stacked = np.concatenate(rows, axis=0)
    # Truncated SVD via numpy. For toy scale this is fine; v0.5 uses sparse SVD.
    U, sigma, V_T = np.linalg.svd(stacked, full_matrices=False)
    k_eff = min(k, V_T.shape[0])
    return RoutingDirections(
        U=U[:, :k_eff],
        sigma=sigma[:k_eff],
        V_T=V_T[:k_eff, :],
        k=k_eff,
        D=D,
        source_hash=_hash_source(stacked),
    )


class RoutingMapper:
    """Cache + lazy-recompute SVD routing directions.

    Use this when the codebook + roles + HYMN-in matrices are stable but may
    change occasionally (e.g., new tokens added). Recomputation is triggered
    when the source-content hash changes.
    """

    def __init__(self, k: int = 64, rebuild_threshold: float = 0.05) -> None:
        """rebuild_threshold reserved for future row-delta-based recompute.
        Currently we recompute on ANY change of the source hash."""
        self.k = k
        self.rebuild_threshold = rebuild_threshold
        self._cached: RoutingDirections | None = None

    def get(
        self,
        codebook_matrix: np.ndarray,
        role_matrix: np.ndarray | None = None,
        W_hymn_in: np.ndarray | None = None,
    ) -> RoutingDirections:
        """Return cached directions if source hash unchanged, else recompute."""
        rows: list[np.ndarray] = [codebook_matrix.astype(np.float32)]
        if role_matrix is not None:
            rows.append(role_matrix.astype(np.float32))
        if W_hymn_in is not None:
            rows.append(W_hymn_in.T.astype(np.float32))
        stacked = np.concatenate(rows, axis=0)
        new_hash = _hash_source(stacked)
        if self._cached is None or self._cached.source_hash != new_hash:
            self._cached = compute_routing_directions(
                codebook_matrix, role_matrix, W_hymn_in, k=self.k
            )
        return self._cached
