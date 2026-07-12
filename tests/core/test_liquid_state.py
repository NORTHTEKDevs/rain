"""Tests for the bipolar LSM + RLS readout."""

import numpy as np
import pytest

from rain.core.liquid_state import LiquidStateMachine


def test_step_changes_state():
    lsm = LiquidStateMachine(input_dim=64, reservoir_dim=32, seed=0)
    s0 = lsm.state.copy()
    lsm.step(np.ones(64, dtype=np.float32))
    assert not np.allclose(lsm.state, s0)


def test_state_shape():
    lsm = LiquidStateMachine(input_dim=128, reservoir_dim=64, seed=1)
    lsm.step(np.zeros(128, dtype=np.float32))
    assert lsm.state.shape == (64,)


def test_input_dim_mismatch_raises():
    lsm = LiquidStateMachine(input_dim=64, reservoir_dim=32, seed=2)
    with pytest.raises(ValueError):
        lsm.step(np.zeros(65, dtype=np.float32))


def test_rls_reduces_residual_over_time():
    """RLS readout learns a repeating sequence; residual should drop to near-zero."""
    rng = np.random.default_rng(42)
    lsm = LiquidStateMachine(input_dim=32, reservoir_dim=128, seed=42)
    # 4-step repeating bipolar pattern -- learnable by reservoir
    pattern = rng.choice([-1.0, 1.0], size=(4, 32)).astype(np.float32)
    seq = np.tile(pattern, (10, 1))  # 40 steps
    residuals = []
    for t in range(len(seq) - 1):
        lsm.step(seq[t])
        r = lsm.update(seq[t + 1])
        residuals.append(r)
    early = float(np.mean(residuals[:5]))
    late = float(np.mean(residuals[-5:]))
    assert late < early, f"residual did not drop: early={early}, late={late}"


def test_reset_zeros_state():
    lsm = LiquidStateMachine(input_dim=16, reservoir_dim=8, seed=0)
    lsm.step(np.ones(16, dtype=np.float32))
    assert np.any(lsm.state != 0)
    lsm.reset()
    assert np.all(lsm.state == 0)
