"""Tests for the HYMN-Plus eval script."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_plus import HymnPlus, HymnPlusConfig  # noqa: E402
from scripts.eval_hymn_plus import _load_checkpoint, evaluate_nll  # noqa: E402


def _save_toy_checkpoint(tmp_path, vocab=None):
    if vocab is None:
        vocab = list("abcdefghij ")
    cfg = HymnPlusConfig(vocab_size=len(vocab), dim=32, n_layers=1, mlp_mult=2, seed=0)
    m = HymnPlus(cfg)
    ckpt = tmp_path / "toy.npz"
    np.savez_compressed(ckpt, **{k: v.detach().cpu().numpy() for k, v in m.state_dict().items()})
    ckpt.with_suffix(".json").write_text(
        json.dumps(
            {
                "arch": "hymn_plus_v1",
                "dim": 32,
                "n_layers": 1,
                "mlp_mult": 2,
                "vocab_size": len(vocab),
                "char_vocab": list(vocab),
                "seed": 0,
            }
        )
    )
    return ckpt, vocab


def test_evaluate_nll_returns_finite(tmp_path):
    ckpt, vocab = _save_toy_checkpoint(tmp_path)
    model, char_to_id, _ = _load_checkpoint(ckpt)
    text = "".join(vocab) * 100  # 1100 chars, all in vocab
    res = evaluate_nll(model, char_to_id, text, batch_size=4, seq_len=32)
    assert "nll_nats_per_char" in res
    assert res["coverage"] == 1.0
    assert 0 < res["nll_nats_per_char"] < 100  # finite
    assert "perplexity" in res
    assert res["uniform_baseline_nll"] == pytest.approx(np.log(len(vocab)), abs=1e-6)


def test_evaluate_skips_unknown_chars(tmp_path):
    ckpt, vocab = _save_toy_checkpoint(tmp_path, list("abc"))
    model, char_to_id, _ = _load_checkpoint(ckpt)
    text = "abcXYZabc" * 200  # 1800 chars, 1/3 unknown
    res = evaluate_nll(model, char_to_id, text, batch_size=4, seq_len=16)
    assert 0.6 < res["coverage"] < 0.7  # ~6/9 in vocab


def test_evaluate_handles_too_short(tmp_path):
    ckpt, vocab = _save_toy_checkpoint(tmp_path)
    model, char_to_id, _ = _load_checkpoint(ckpt)
    res = evaluate_nll(model, char_to_id, "ab", batch_size=1, seq_len=32)
    assert "error" in res
