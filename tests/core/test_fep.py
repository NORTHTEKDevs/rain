# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the bipolar FEP low-rank A."""

import numpy as np
import pytest
from rain.core.fep import LowRankA


def test_predict_shape():
    a = LowRankA(D=128, R=8, seed=0)
    s = np.ones(128, dtype=np.float32)
    out = a.predict(s)
    assert out.shape == (128,)


def test_rank_preserved_after_update():
    a = LowRankA(D=64, R=16, seed=1)
    assert a.rank() == 16
    s = np.ones(64, dtype=np.float32)
    t = np.full(64, 0.5, dtype=np.float32)
    a.update(s, t, alpha=0.1)
    assert a.rank() == 16


def test_update_reduces_residual_on_repeated_target():
    """Repeating the same (s, t) pair should drive residual down."""
    rng = np.random.default_rng(7)
    a = LowRankA(D=64, R=32, seed=7)
    s = rng.standard_normal(64).astype(np.float32)
    t = rng.standard_normal(64).astype(np.float32)
    residuals = []
    for _ in range(100):
        residuals.append(a.update(s, t, alpha=0.05))
    early = float(np.mean(residuals[:10]))
    late = float(np.mean(residuals[-10:]))
    assert late < early, f"early {early} late {late}"


def test_shape_mismatch_raises():
    a = LowRankA(D=32, R=4, seed=0)
    with pytest.raises(ValueError):
        a.predict(np.zeros(33, dtype=np.float32))
    with pytest.raises(ValueError):
        a.update(np.zeros(32, dtype=np.float32), np.zeros(33, dtype=np.float32))


def test_predict_is_deterministic_with_same_seed():
    a1 = LowRankA(D=32, R=4, seed=99)
    a2 = LowRankA(D=32, R=4, seed=99)
    s = np.ones(32, dtype=np.float32)
    assert np.allclose(a1.predict(s), a2.predict(s))
