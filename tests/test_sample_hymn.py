# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the HYMN sampling helpers (no checkpoint required)."""

import numpy as np

from scripts.sample_hymn import _sample_from_logits


def test_sample_argmax_at_temperature_zero():
    """temperature <= 0 -> deterministic argmax."""
    logits = np.array([0.1, 0.5, 0.9, 0.3, 0.7], dtype=np.float64)
    rng = np.random.default_rng(0)
    assert _sample_from_logits(logits, temperature=0.0, rng=rng) == 2


def test_sample_with_top_k_excludes_low_logits():
    """top_k=1 forces argmax even with positive temperature."""
    logits = np.array([0.1, 0.5, 0.9, 0.3, 0.7], dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert _sample_from_logits(logits, temperature=0.5, rng=rng, top_k=1) == 2


def test_repetition_penalty_avoids_argmax_loop():
    """With strong repetition penalty + argmax, the same id should NOT be picked
    twice in a row when there's an alternative."""
    logits = np.array([0.5, 1.0, 0.3], dtype=np.float64)
    rng = np.random.default_rng(0)
    first = _sample_from_logits(
        logits, temperature=0.0, rng=rng, recent_ids=[], repetition_penalty=2.0
    )
    assert first == 1  # argmax with empty history
    # Now id 1 is in history; with penalty=2.0 its logit becomes 0.5,
    # tied with id 0's 0.5; argmax picks the earlier index -> 0.
    second = _sample_from_logits(
        logits, temperature=0.0, rng=rng, recent_ids=[1], repetition_penalty=2.0
    )
    assert second != 1


def test_repetition_penalty_one_is_a_noop():
    logits = np.array([0.5, 1.0, 0.3], dtype=np.float64)
    rng = np.random.default_rng(0)
    plain = _sample_from_logits(
        logits, temperature=0.0, rng=rng, recent_ids=[], repetition_penalty=1.0
    )
    with_history = _sample_from_logits(
        logits, temperature=0.0, rng=rng, recent_ids=[1], repetition_penalty=1.0
    )
    assert plain == with_history == 1
