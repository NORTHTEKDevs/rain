# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Ported from Hyperion design/RESONATOR-FINDINGS.md (Findings 1-5), adapted to
# RAIN's MAP-VSA convention (bipolar, elementwise bind, unbind == bind).
"""Resonator network + resonant explaining-away decode for RAIN.

Two capabilities RAIN's bind/unbind substrate lacked:

1. resonator_decode -- factorize a bound PRODUCT  s = x_1 * ... * x_F  into the
   identities of F unknown factors, each one entry of a known codebook, by an
   iterative attractor dynamic (Frady, Kent, Olshausen & Sommer 2020). This is the
   binding-problem solver: read structured state when the factors are NOT known.
   Brute force is M^F; resonance is O(F*M*D) per iteration.

2. resonant_extract -- joint, interference-cancelling readout of a superposition
   of role->value bindings  sum_i bind(role_i, val_i). RAIN's CompositionalReasoner
   currently decodes each slot independently (greedy), which fails once cross-talk
   from the other slots swamps the signal. Re-reading each slot after subtracting
   the current reconstruction of the others recovers materially more slots at a
   given dimension (Hyperion Finding 5: ~4-5x longer decodable sequences).

Capacity rule of thumb (Hyperion Finding 1): resonance is reliable for a small
number of simultaneously-bound factors and scales cleanly with dimension D.
"""

from __future__ import annotations

import numpy as np


def _sim(codebook: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise dot of (M,D) codebook with (D,) query."""
    return codebook.astype(np.float64) @ q.astype(np.float64)


def resonator_decode(product: np.ndarray, codebooks: list[np.ndarray],
                     max_iters: int = 50) -> list[int]:
    """Recover factor indices for product = x_1 * ... * x_F (elementwise bind).

    codebooks: list of F arrays, each (M_f, D) bipolar. Returns F indices.
    """
    F = len(codebooks)
    D = product.shape[0]
    est = [np.sign(cb.sum(0) + (cb.sum(0) == 0)) for cb in codebooks]  # superposition init
    idx = [-1] * F
    for _ in range(max_iters):
        new_est: list[np.ndarray] = []
        new_idx: list[int] = []
        for i in range(F):
            others = np.ones(D, dtype=np.float64)
            for j in range(F):
                if j != i:
                    others = others * est[j]
            ci = product.astype(np.float64) * others          # unbind (elementwise)
            sims = _sim(codebooks[i], ci)                      # (M_f,)
            recon = sims @ codebooks[i].astype(np.float64)     # weighted superposition
            new_est.append(np.sign(recon + (recon == 0)))
            new_idx.append(int(sims.argmax()))
        if new_idx == idx:
            break
        est, idx = new_est, new_idx
    return idx


def greedy_extract(superposition: np.ndarray, roles: np.ndarray,
                   codebook: np.ndarray) -> list[int]:
    """Per-slot independent decode (RAIN's current behaviour) for comparison.

    superposition: (D,) real sum of bind(role_i, val_i). roles: (S,D). codebook:(M,D).
    """
    S = roles.shape[0]
    return [int(_sim(codebook, superposition * roles[i]).argmax()) for i in range(S)]


def resonant_extract(superposition: np.ndarray, roles: np.ndarray,
                     codebook: np.ndarray, max_iters: int = 25) -> list[int]:
    """Joint explaining-away decode of sum_i bind(role_i, val_i).

    Re-reads each slot after subtracting the reconstruction of all other slots,
    iterated to a fixed point. Returns S indices into codebook.
    """
    S = roles.shape[0]
    idx = greedy_extract(superposition, roles, codebook)       # init
    sup = superposition.astype(np.float64)
    for _ in range(max_iters):
        recon = np.stack([roles[i].astype(np.float64) * codebook[idx[i]].astype(np.float64)
                          for i in range(S)])                  # (S,D) per-slot bind
        total = recon.sum(0)
        new_idx: list[int] = []
        for i in range(S):
            residual = sup - (total - recon[i])                # remove other slots
            new_idx.append(int(_sim(codebook, residual * roles[i].astype(np.float64)).argmax()))
        if new_idx == idx:
            break
        idx = new_idx
    return idx
