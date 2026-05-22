# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
import numpy as np
from rain.core.relational import Codebook
from rain.core.bigram import BigramMemory


def test_add_and_query_finds_added_bigram():
    cb = Codebook(vocab_size=20, dim=512, seed=0)
    bm = BigramMemory(cb, order=2)
    bm.add(["the", "quick"], "brown")
    bm.add(["the", "lazy"], "dog")
    candidates = ["brown", "dog", "fox", "house", "tree"]
    result = bm.query(["the", "quick"], candidates, top_k=3)
    assert len(result) == 3
    # "brown" should be in top-2 (signal/noise margin at 20 vocab/512 dim)
    top_tokens = [t for t, _ in result[:2]]
    assert "brown" in top_tokens


def test_empty_memory_returns_empty():
    cb = Codebook(vocab_size=10, dim=256, seed=0)
    bm = BigramMemory(cb, order=2)
    assert bm.query(["a", "b"], ["c", "d"]) == []
