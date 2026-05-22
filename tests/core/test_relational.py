# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for bipolar 10K-dim VSA primitives."""

import numpy as np
import pytest
from rain.core.relational import Codebook, bind, bundle, unbind

D = 10000

def test_codebook_values_are_bipolar():
    cb = Codebook(vocab_size=100, dim=D, seed=42)
    hv = cb.vector("hello")
    assert hv.shape == (D,)
    assert set(np.unique(hv).tolist()) <= {-1, 1}

def test_bind_is_self_inverse_for_bipolar():
    cb = Codebook(vocab_size=10, dim=D, seed=0)
    a = cb.vector("a")
    b = cb.vector("b")
    bound = bind(a, b)
    recovered = unbind(bound, a)
    # Cosine similarity to b should be high
    sim = (recovered @ b) / (np.linalg.norm(recovered) * np.linalg.norm(b))
    assert sim > 0.95

def test_bundle_preserves_membership():
    cb = Codebook(vocab_size=10, dim=D, seed=0)
    items = [cb.vector(s) for s in ["a", "b", "c"]]
    bundle_hv = bundle(items)
    for item in items:
        sim = (bundle_hv @ item) / D
        assert sim > 0.3  # bundling preserves approximate membership

def test_seeding_is_deterministic():
    cb1 = Codebook(vocab_size=5, dim=D, seed=123)
    cb2 = Codebook(vocab_size=5, dim=D, seed=123)
    assert np.array_equal(cb1.vector("x"), cb2.vector("x"))
