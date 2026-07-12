"""Tests for the feature-based char warm-start."""

import numpy as np

from rain.core.relational import Codebook
from rain.train.warm_start_chars import (
    _char_features,
    _projection_matrix,
    warm_start_chars,
)


def test_features_distinguish_categories():
    a_feats = _char_features("a", 0.05)
    z_feats = _char_features("Z", 0.001)
    d_feats = _char_features("3", 0.01)
    s_feats = _char_features(" ", 0.15)
    # Lowercase 'a': is_letter=1, is_lower=1, is_upper=0, is_vowel=1
    assert a_feats[0] == 1.0 and a_feats[1] == 1.0 and a_feats[6] == 1.0
    # Uppercase 'Z': is_letter=1, is_upper=1, is_consonant=1, is_vowel=0
    assert z_feats[0] == 1.0 and z_feats[2] == 1.0 and z_feats[7] == 1.0
    # Digit '3': is_letter=0, is_digit=1
    assert d_feats[0] == 0.0 and d_feats[3] == 1.0
    # Space: is_space=1
    assert s_feats[5] == 1.0


def test_projection_is_deterministic_for_same_seed():
    a = _projection_matrix(seed=7, target_dim=64)
    b = _projection_matrix(seed=7, target_dim=64)
    assert np.array_equal(a, b)


def test_warm_start_produces_bipolar_vectors():
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    stats = warm_start_chars(cb, "hello world", seed=42)
    assert stats["chars_seeded"] >= 5  # at least h, e, l, o, space, w, r, d
    # Every seeded vector should be bipolar.
    for c in "hello world":
        v = cb.vector(c)
        assert v.shape == (128,)
        assert set(np.unique(v).tolist()) <= {-1, 1}


def test_within_letters_more_similar_than_across_categories():
    """The whole point of the warm-start: similar-category chars cluster."""
    cb = Codebook(vocab_size=128, dim=512, seed=0)
    text = "the quick brown fox jumps over 0123456789 the lazy dog"
    stats = warm_start_chars(cb, text, seed=42)
    # Within-letters should be much more similar than letters-to-digits.
    assert stats["mean_cos_within_letters"] > 0.3
    assert stats["mean_cos_letters_to_digits"] < 0.3
    assert stats["mean_cos_within_letters"] > stats["mean_cos_letters_to_digits"]


def test_warm_start_seeds_change_projection():
    cb1 = Codebook(vocab_size=64, dim=64, seed=0)
    cb2 = Codebook(vocab_size=64, dim=64, seed=0)
    warm_start_chars(cb1, "abc", seed=1)
    warm_start_chars(cb2, "abc", seed=2)
    # Different seeds -> different vectors.
    assert not np.array_equal(cb1.vector("a"), cb2.vector("a"))
