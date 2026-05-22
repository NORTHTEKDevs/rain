# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
import pytest
from rain.core.relational import Codebook
from rain.core.efe import MultiSignalEFEDecoder, EpistemicClass, DEFAULT_WEIGHTS, _rank_normalize


def test_rank_normalize_orders_correctly():
    out = _rank_normalize([("a", 1.0), ("b", 5.0), ("c", 3.0)])
    # Sorted by score desc: b(5) -> 1.0, c(3) -> 0.5, a(1) -> 0.0
    d = dict(out)
    assert d["b"] == 1.0
    assert d["a"] == 0.0


def test_decoder_returns_unknown_with_no_sources_active():
    cb = Codebook(vocab_size=10, dim=128, seed=0)
    decoder = MultiSignalEFEDecoder(codebook=cb)  # no sources hooked
    state = np.zeros(128, dtype=np.float32)
    result = decoder.decode(state, vocab=["a", "b"])
    assert result.chosen_token is None
    assert result.epistemic == EpistemicClass.UNKNOWN


def test_decoder_default_weights_are_per_design_doc():
    assert DEFAULT_WEIGHTS["hymn"] == 4.0
    assert DEFAULT_WEIGHTS["kb"] == 3.0
    assert DEFAULT_WEIGHTS["bigram"] == 0.5
