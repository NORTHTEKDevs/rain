# tests/integration/test_phase1_setup.py

import numpy as np

from rain.core.knowledge_base import ShardedKB
from rain.core.relational import Codebook
from rain.tokenize.bpe import BPETokenizer
from scripts.bootstrap_warm_start import warm_start_from_vectors


def test_end_to_end_phase1_setup():
    tok = BPETokenizer(vocab_size=256)
    tok.train(["the quick brown fox", "the lazy dog runs"])

    cb = Codebook(vocab_size=256, dim=10000, seed=42)
    fake_vecs = {"hello": np.random.RandomState(0).randn(300)}
    n = warm_start_from_vectors(cb, fake_vecs)
    assert n == 1

    kb = ShardedKB(num_shards=8, dim=10000, seed=42)
    kb.write("dog", "is_a", "animal")
    assert kb.query("dog", "is_a") == "animal"
