# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the bipolar Tsetlin Machine port."""

import numpy as np
import pytest
from rain.core.tsetlin import TsetlinMachine


def test_initial_state_no_clauses_fire():
    tm = TsetlinMachine(num_classes=3, num_clauses_per_class=4, num_features=64, seed=0)
    state = np.ones(64, dtype=np.int8)
    scores = tm.vote(state)
    assert scores.shape == (3,)
    # All clauses empty -> no firing -> zero scores
    assert np.all(scores == 0)


def test_vote_shape_and_dtype():
    tm = TsetlinMachine(num_classes=2, num_clauses_per_class=8, num_features=32, seed=42)
    state = np.where(np.arange(32) % 2 == 0, 1, -1).astype(np.int8)
    scores = tm.vote(state)
    assert scores.shape == (2,)


def test_feedback_makes_target_class_clauses_fire_more():
    rng = np.random.default_rng(7)
    tm = TsetlinMachine(num_classes=2, num_clauses_per_class=4, num_features=32, seed=7)
    state = rng.choice([-1, 1], size=32).astype(np.int8)
    # Run feedback a few times targeting class 0
    for _ in range(50):
        tm.feedback(state, target_class=0)
    scores = tm.vote(state)
    # Class 0 should have a positive score after feedback
    assert scores[0] > scores[1], f"scores: {scores}"


def test_predict_returns_argmax_class():
    tm = TsetlinMachine(num_classes=3, num_clauses_per_class=2, num_features=16, seed=0)
    state = np.ones(16, dtype=np.int8)
    # No clauses fire -> all scores zero -> argmax = 0
    assert tm.predict(state) == 0


def test_invalid_state_shape_raises():
    tm = TsetlinMachine(num_classes=2, num_clauses_per_class=2, num_features=16, seed=0)
    with pytest.raises(ValueError):
        tm.vote(np.ones(17, dtype=np.int8))
