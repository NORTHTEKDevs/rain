# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""End-to-end Phase 1 + Phase 3 bootstrap integration test."""

import json

import numpy as np

from rain.train.bootstrap import RainBootstrap, bootstrap_phase1


def test_bootstrap_composes_all_components(tmp_path):
    corpus = "the quick brown fox jumps over the lazy dog" * 50
    kb_path = tmp_path / "seed.jsonl"
    kb_path.write_text('{"s":"rome","r":"capital_of","o":"italy"}\n')
    fake_vecs = {"hello": np.random.RandomState(0).randn(300)}

    boot = bootstrap_phase1(
        corpus_texts=[corpus],
        kb_jsonl_path=str(kb_path),
        warm_start_vectors=fake_vecs,
        D=512,
        vocab_size=64,
        num_shards=4,
        tsetlin_classes=2,
        tsetlin_clauses_per_class=4,
        lsm_reservoir=32,
        fep_rank=8,
        routing_k=4,
        bpe_vocab=256,
        seed=0,
    )

    assert isinstance(boot, RainBootstrap)
    assert boot.codebook.dim == 512
    assert boot.kb.dim == 512
    assert boot.kb.query("rome", "capital_of") == "italy"
    assert boot.fep.D == 512
    assert boot.lsm.input_dim == 512
    assert boot.tsetlin.num_features == 512
    assert boot.beam_adapter.D == 512


def test_bootstrap_summary_is_json_serializable():
    boot = bootstrap_phase1(
        corpus_texts=["hello world " * 20],
        D=128, vocab_size=32, num_shards=2, tsetlin_classes=2,
        tsetlin_clauses_per_class=2, lsm_reservoir=8, fep_rank=4,
        routing_k=4, bpe_vocab=64, seed=0,
    )
    summary = boot.summary()
    # Must JSON serialize without errors
    s = json.dumps(summary)
    assert "config" in s


def test_components_are_independently_callable():
    """Bootstrap returns components that can be used independently."""
    boot = bootstrap_phase1(
        corpus_texts=["abc def " * 30],
        D=64, vocab_size=16, num_shards=2, tsetlin_classes=2,
        tsetlin_clauses_per_class=2, lsm_reservoir=8, fep_rank=4,
        routing_k=2, bpe_vocab=64, seed=0,
    )
    # KB write+query
    boot.kb.write("a", "r", "b")
    assert boot.kb.query("a", "r") == "b"
    # FEP predict
    s = np.ones(64, dtype=np.float32)
    pred = boot.fep.predict(s)
    assert pred.shape == (64,)
    # LSM step
    boot.lsm.step(s)
    assert boot.lsm.state.shape == (8,)
    # Tsetlin vote
    state_i16 = np.where(np.arange(64) % 2 == 0, 1, -1).astype(np.int8)
    scores = boot.tsetlin.vote(state_i16)
    assert scores.shape == (2,)
    # Tokenizer encode
    ids = boot.tokenizer.encode("abc")
    assert len(ids) > 0
