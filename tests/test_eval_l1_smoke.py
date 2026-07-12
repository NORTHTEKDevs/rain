"""Smoke test for the L1 Tiny Shakespeare evaluator. Verifies the scaffold
runs end-to-end on a small randomly-init checkpoint. Expected: val_loss is
ABOVE threshold (random init doesn't pass), so pass=False -- that is correct
scaffold behavior."""

import json

import numpy as np

from evals.tier2_llm_parity.tiny_shakespeare import L1_PASS_THRESHOLD_HV_MSE, evaluate
from rain.train.checkpoint import HymnCheckpointMetadata, save_checkpoint


def test_l1_runs_end_to_end_on_random_init(tmp_path):
    # Build a tiny random-init checkpoint
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((64, 32)).astype(np.float32)
    W2 = rng.standard_normal((32, 64)).astype(np.float32)
    meta = HymnCheckpointMetadata(
        in_dim=64,
        hidden_dim=32,
        out_dim=64,
        steps=0,
        lr=0.01,
        seed=42,
        final_loss=None,
        initial_loss=None,
    )
    save_checkpoint(tmp_path / "random_ckpt", W1, W2, meta)

    # Build a tiny corpus
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("the quick brown fox jumps over the lazy dog" * 20)

    # Run eval
    result = evaluate(str(tmp_path / "random_ckpt"), str(corpus_path), n_eval_chars=50)
    assert result["benchmark"] == "L1_tiny_shakespeare"
    assert result["metric_type"] == "hv_mse"
    assert result["threshold"] == L1_PASS_THRESHOLD_HV_MSE
    assert result["nll_threshold_applicable"] is False
    assert isinstance(result["hv_mse_loss"], float)
    assert "pass" in result
    assert result["checkpoint_metadata"]["in_dim"] == 64
    # Random init should NOT pass the threshold -- that's correct scaffold behavior.
    # We only verify the plumbing produced a sensible result, not the value.


def test_l1_writes_json_via_cli(tmp_path):
    """The CLI path is exercised via the same evaluate() function -- verify JSON serializability."""
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((32, 16)).astype(np.float32)
    W2 = rng.standard_normal((16, 32)).astype(np.float32)
    meta = HymnCheckpointMetadata(
        in_dim=32,
        hidden_dim=16,
        out_dim=32,
        steps=0,
        lr=0.01,
        seed=42,
        final_loss=None,
        initial_loss=None,
    )
    save_checkpoint(tmp_path / "ckpt", W1, W2, meta)
    corpus_path = tmp_path / "c.txt"
    corpus_path.write_text("hello world " * 50)
    result = evaluate(str(tmp_path / "ckpt"), str(corpus_path), n_eval_chars=20)
    # Must JSON-serialize cleanly
    json.dumps(result)
