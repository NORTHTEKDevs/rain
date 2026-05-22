# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for rain.spine.HymnModel (Python wrapper over rain-rs HYMN)."""

import numpy as np
import pytest

try:
    from rain.spine import HymnModel, _RUST_AVAILABLE
except ImportError:
    _RUST_AVAILABLE = False


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="rain._rust not built")
def test_forward_output_shape_and_bipolar():
    model = HymnModel(in_dim=32, hidden_dim=16, out_dim=32, seed=7)
    rng = np.random.default_rng(0)
    state = rng.choice([-1, 1], size=32).astype(np.int16)
    input_ = rng.choice([-1, 1], size=32).astype(np.int16)
    out = model.forward(state, input_)
    assert out.shape == (32,)
    assert set(np.unique(out).tolist()) <= {-1, 1}


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="rain._rust not built")
def test_forward_deterministic_with_seed():
    m1 = HymnModel(in_dim=16, hidden_dim=8, out_dim=16, seed=42)
    m2 = HymnModel(in_dim=16, hidden_dim=8, out_dim=16, seed=42)
    state = np.ones(16, dtype=np.int16)
    input_ = -np.ones(16, dtype=np.int16)
    assert np.array_equal(m1.forward(state, input_), m2.forward(state, input_))
