"""Tests for rain.core.resonator (ported from Hyperion Findings 1-5, MAP-VSA).

Covers: (1) resonator factorization of an unknown product; (2) resonant
explaining-away beating greedy per-slot decode under cross-talk; (3) the
CompositionalReasoner.extract_all integration recovering many slots where the
per-slot extract() collapses.
"""

from __future__ import annotations

import numpy as np

from rain.cognition.compose import CompositionalReasoner
from rain.core.relational import Codebook
from rain.core.resonator import greedy_extract, resonant_extract, resonator_decode


def _hvs(rng, n, D):
    return rng.choice([-1, 1], size=(n, D)).astype(np.int16)


def test_resonator_factorizes_unknown_product():
    """F=3 factors from M=10 codebooks at D=2048 -> near-perfect recovery."""
    rng = np.random.default_rng(0)
    D, M, F, T = 2048, 10, 3, 60
    cbs = [_hvs(rng, M, D) for _ in range(F)]
    ok = 0
    for _ in range(T):
        true = [int(rng.integers(M)) for _ in range(F)]
        prod = np.ones(D, dtype=np.int16)
        for f in range(F):
            prod = (prod * cbs[f][true[f]]).astype(np.int16)
        ok += (resonator_decode(prod, cbs) == true)
    assert ok / T >= 0.9


def test_restarts_improve_factorization_at_capacity():
    """Near the factor-capacity edge (F=4), random restarts + best-reconstruction
    selection escape limit cycles and roughly double single-pass success."""
    rng = np.random.default_rng(5)
    D, M, F, T = 4096, 10, 4, 40
    cbs = [_hvs(rng, M, D) for _ in range(F)]
    single = multi = 0
    for _ in range(T):
        true = [int(rng.integers(M)) for _ in range(F)]
        prod = np.ones(D, dtype=np.int16)
        for f in range(F):
            prod = (prod * cbs[f][true[f]]).astype(np.int16)
        single += (resonator_decode(prod, cbs, restarts=1) == true)
        multi += (resonator_decode(prod, cbs, restarts=6) == true)
    assert multi > single        # restarts help at the capacity edge
    assert multi / T >= 0.45     # reach a usable rate where single-shot is ~0.28


def test_resonant_extract_beats_greedy_under_crosstalk():
    """At D=256, S=32 role->value bindings, greedy per-slot decode collapses while
    resonant explaining-away recovers all slots."""
    rng = np.random.default_rng(1)
    D, M, S, T = 256, 12, 32, 30
    g_ok = r_ok = 0
    for _ in range(T):
        roles = _hvs(rng, S, D)
        cb = _hvs(rng, M, D)
        vals = [int(rng.integers(M)) for _ in range(S)]
        sup = np.zeros(D, dtype=np.float64)
        for i in range(S):
            sup += roles[i] * cb[vals[i]]
        g_ok += (greedy_extract(sup, roles, cb) == vals)
        r_ok += (resonant_extract(sup, roles, cb) == vals)
    assert g_ok / T <= 0.3        # greedy collapses under cross-talk
    assert r_ok / T >= 0.9        # resonance recovers
    assert r_ok > g_ok


def test_compositional_reasoner_extract_all_scales():
    """CompositionalReasoner.extract_all (resonant) recovers many slots where the
    per-slot extract() on the signed bundle fails."""
    cb = Codebook(vocab_size=200, dim=256, seed=0)
    S, M = 28, 10
    slots = [f"s{i}" for i in range(S)]
    candidates = [f"v{j}" for j in range(M)]
    reasoner = CompositionalReasoner(cb, slots)
    rng = np.random.default_rng(2)

    n_trials, all_correct_resonant, all_correct_greedy = 8, 0, 0
    for _ in range(n_trials):
        assign = {s: candidates[int(rng.integers(M))] for s in slots}
        # resonant joint decode from the unsigned superposition
        pred = reasoner.extract_all(reasoner.compose_sum(assign), slots, candidates)
        all_correct_resonant += int(all(pred[s] == assign[s] for s in slots))
        # per-slot greedy decode from the signed bundle (current behaviour)
        bundle = reasoner.compose(assign)
        greedy_ok = all(reasoner.extract(bundle, s, candidates) == assign[s] for s in slots)
        all_correct_greedy += int(greedy_ok)
    assert all_correct_resonant >= n_trials - 1   # ~all trials fully recovered
    assert all_correct_resonant > all_correct_greedy
