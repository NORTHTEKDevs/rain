# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
from rain.core.relational import Codebook
from scripts.bootstrap_warm_start import warm_start_from_vectors


def test_warm_start_overrides_random():
    cb = Codebook(vocab_size=3, dim=10000, seed=0)
    original = cb.vector("hello").copy()
    fake_vectors = {"hello": np.random.RandomState(99).randn(300)}
    warm_start_from_vectors(cb, fake_vectors)
    new = cb.vector("hello")
    # Same set of -1/+1
    assert set(np.unique(new).tolist()) <= {-1, 1}
    # Different from random init
    assert not np.array_equal(new, original)
