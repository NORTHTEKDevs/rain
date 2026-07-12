"""Tests for SOFAR-on-VSA beam-steering adapter."""

import numpy as np
import pytest

from rain.routing.beam import (
    BeamConfig,
    BeamMode,
    BeamSteeringAdapter,
    _energy_normalize,
    _envelope,
)
from rain.routing.mapper import compute_routing_directions

D = 64
K = 8


def _mk_directions(rng):
    cb = rng.choice([-1, 1], size=(D, D)).astype(np.float32)
    return compute_routing_directions(cb, role_matrix=None, W_hymn_in=None, k=K)


def test_envelopes_are_correct_length():
    for mode in (BeamMode.FOCUS, BeamMode.SWEEP, BeamMode.TRACK):
        env = _envelope(mode, K, 2.0, 1.0)
        assert env.shape == (K,)


def test_energy_normalization_yields_sqrt_k_norm():
    env = _envelope(BeamMode.FOCUS, K, 4.0, 2.0)
    nrm = _energy_normalize(env)
    actual_norm = float(np.linalg.norm(nrm))
    assert abs(actual_norm - np.sqrt(K)) < 1e-4


def test_identity_at_init():
    rng = np.random.default_rng(0)
    directions = _mk_directions(rng)
    adapter = BeamSteeringAdapter(k=K, D=D, lora_rank=4, seed=0)
    state = rng.choice([-1, 1], size=D).astype(np.float32)
    routed = adapter.apply(state, directions, BeamConfig(BeamMode.FOCUS, 2.0, 1.0))
    # With beam_gate=0 AND lora_up all zero, output must equal input exactly
    assert np.allclose(routed, state)


def test_focus_changes_state_when_gate_open():
    rng = np.random.default_rng(1)
    directions = _mk_directions(rng)
    adapter = BeamSteeringAdapter(k=K, D=D, lora_rank=4, seed=1)
    adapter.beam_gate = np.float32(1.0)
    # Random non-zero lora_up so we exit the identity-at-init guard
    adapter.lora_up = rng.standard_normal((4, K)).astype(np.float32) * 0.01
    state = rng.choice([-1, 1], size=D).astype(np.float32)
    routed = adapter.apply(state, directions, BeamConfig(BeamMode.FOCUS, 2.0, 1.0))
    assert not np.allclose(routed, state)


def test_modes_produce_different_outputs():
    rng = np.random.default_rng(2)
    directions = _mk_directions(rng)
    adapter = BeamSteeringAdapter(k=K, D=D, lora_rank=4, seed=2)
    adapter.beam_gate = np.float32(1.0)
    adapter.lora_up = rng.standard_normal((4, K)).astype(np.float32) * 0.01
    state = rng.choice([-1, 1], size=D).astype(np.float32)
    f = adapter.apply(state, directions, BeamConfig(BeamMode.FOCUS, 2.0, 1.0))
    s = adapter.apply(state, directions, BeamConfig(BeamMode.SWEEP, 2.0, 2.0))
    t = adapter.apply(state, directions, BeamConfig(BeamMode.TRACK, 2.0, 1.0))
    assert not np.allclose(f, s)
    assert not np.allclose(s, t)
    assert not np.allclose(f, t)


def test_dim_mismatch_raises():
    rng = np.random.default_rng(3)
    directions = _mk_directions(rng)
    adapter = BeamSteeringAdapter(k=K, D=D + 1, lora_rank=4, seed=0)
    state = np.zeros(D + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        adapter.apply(state, directions, BeamConfig(BeamMode.FOCUS, 0.0, 1.0))
