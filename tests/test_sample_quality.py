# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the sample-quality eval primitives."""

import pytest

pytest.importorskip("torch")

from scripts.sample_quality import (  # noqa: E402
    _distinct_ngram_ratio,
    _unique_window_pct,
    _verbatim_overlap_pct,
)


def test_verbatim_overlap_exact_match_is_one():
    corpus = "the quick brown fox jumps over the lazy dog and runs again"
    # sample is a contiguous slice from corpus -> every window matches
    sample = corpus[5 : 5 + 64]
    pct = _verbatim_overlap_pct(sample, corpus, window=16)
    assert pct == 1.0


def test_verbatim_overlap_no_match_is_zero():
    corpus = "the quick brown fox jumps over the lazy dog"
    sample = "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"
    assert _verbatim_overlap_pct(sample, corpus, window=8) == 0.0


def test_verbatim_overlap_handles_short_sample():
    assert _verbatim_overlap_pct("abc", "longer corpus here", window=10) == 0.0


def test_distinct_ngram_repetitive_text_is_low():
    """A sample that says 'abab...' has very few distinct bigrams."""
    s = "abab" * 50
    ratio = _distinct_ngram_ratio(s, 2)
    # Only "ab", "ba" distinct -> 2 / (len-1)
    assert ratio < 0.05


def test_distinct_ngram_varied_text_is_high():
    s = "the quick brown fox jumps over the lazy dog while running"
    ratio = _distinct_ngram_ratio(s, 3)
    assert ratio > 0.5


def test_unique_window_pct_all_same_samples_is_low():
    samples = ["abcdefghijklmnop"] * 4
    pct = _unique_window_pct(samples, window=8)
    # All 4 samples produce same windows -> 4x duplication -> 1/4 unique
    assert pct == pytest.approx(0.25, abs=0.05)


def test_unique_window_pct_diverse_samples_is_high():
    samples = [
        "aaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbb",
        "cccccccccccccccccccccc",
        "dddddddddddddddddddddd",
    ]
    pct = _unique_window_pct(samples, window=8)
    # Each sample has same single window repeated -> 4 distinct windows
    # total / total_count. Each sample has ~14 windows of same.
    # Roughly 4 / (4 * ~15) = ~0.06
    assert pct < 0.2
