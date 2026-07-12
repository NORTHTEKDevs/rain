"""Tests for the SOFAR-on-VSA Walsh-Hadamard band encoder."""

import numpy as np

from rain.routing.encoder import FrequencyBands, encode, reconstruct


def test_encode_returns_three_bands_with_correct_shape():
    rng = np.random.default_rng(0)
    state = rng.choice([-1, 1], size=256).astype(np.int16)
    bands = encode(state)
    assert isinstance(bands, FrequencyBands)
    assert bands.s_L.shape == (256,)
    assert bands.s_M.shape == (256,)
    assert bands.s_H.shape == (256,)
    assert bands.D == 256


def test_round_trip_recovers_input_within_tolerance():
    rng = np.random.default_rng(1)
    state = rng.choice([-1, 1], size=256).astype(np.int16)
    bands = encode(state)
    recovered = reconstruct(bands)
    # Within +/- 1 element-wise for bipolar input
    err = np.abs(recovered - state.astype(np.float32))
    assert err.max() < 1e-3, f"max recon err: {err.max()}"


def test_round_trip_works_for_non_power_of_2_dim():
    # D = 10000 is not a power of 2 — verify padding + truncation work
    rng = np.random.default_rng(2)
    state = rng.choice([-1, 1], size=10000).astype(np.int16)
    bands = encode(state)
    recovered = reconstruct(bands)
    assert recovered.shape == (10000,)
    err = np.abs(recovered - state.astype(np.float32))
    assert err.max() < 1e-2


def test_zero_input_gives_zero_bands():
    state = np.zeros(128, dtype=np.int16)
    bands = encode(state)
    assert np.all(bands.s_L == 0)
    assert np.all(bands.s_M == 0)
    assert np.all(bands.s_H == 0)


def test_bands_are_independent():
    # Different bands should respond to different "frequency" patterns.
    # A constant signal (all +1) should land mostly in the LOW band.
    state = np.ones(256, dtype=np.int16)
    bands = encode(state)
    low_energy = float(np.sum(bands.s_L**2))
    mid_energy = float(np.sum(bands.s_M**2))
    high_energy = float(np.sum(bands.s_H**2))
    assert low_energy > mid_energy, f"low={low_energy} mid={mid_energy}"
    assert low_energy > high_energy, f"low={low_energy} high={high_energy}"
