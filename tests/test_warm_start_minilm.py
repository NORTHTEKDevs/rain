"""Tests for the MiniLM warm-start path.

sentence-transformers is an optional heavy dep -- skip when missing so CI
doesn't have to install ~500 MB of HF cache.
"""

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from rain.core.relational import Codebook  # noqa: E402
from rain.train.warm_start_minilm import (  # noqa: E402
    encode_tokens,
    warm_start_chars_minilm,
)


def test_encode_tokens_returns_per_token_vectors():
    out = encode_tokens(["a", "b", "hello"])
    assert set(out.keys()) == {"a", "b", "hello"}
    for v in out.values():
        assert v.dtype == np.float32
        assert v.shape == (384,)


def test_warm_start_chars_minilm_overwrites_codebook():
    cb = Codebook(vocab_size=256, dim=512, seed=0)
    text = "abcabcabc 123"
    stats = warm_start_chars_minilm(cb, text)
    assert stats["chars_seeded"] > 0
    assert stats["embedding_dim"] == 384.0
    # Each unique char in `text` should now have a bipolar entry in the cache.
    for c in set(text):
        v = cb._cache[c]
        assert v.shape == (512,)
        assert set(np.unique(v)).issubset({-1, 1})


def test_minilm_clusters_letters_more_than_letters_vs_digits():
    """Sanity: letters should be more similar to each other than to digits."""
    cb = Codebook(vocab_size=256, dim=1024, seed=0)
    text = "abcdefghijklmnopqrstuvwxyz 0123456789"
    stats = warm_start_chars_minilm(cb, text)
    # The MiniLM-derived signature gives within-letter cosine higher than
    # letters-to-digits cosine. The bipolar projection keeps the ordering.
    assert stats["mean_cos_within_letters"] > stats["mean_cos_letters_to_digits"]
