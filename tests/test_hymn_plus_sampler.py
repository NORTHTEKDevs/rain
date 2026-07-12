"""Tests for the HYMN-Plus sampler adapter."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rain.cognition.hymn_plus_sampler import HymnPlusSampler  # noqa: E402
from rain.core.hymn_plus import HymnPlus, HymnPlusConfig  # noqa: E402


def _save_tiny_checkpoint(tmp_path, vocab):
    """Train a 1-step toy model and save it in the format HymnPlusSampler expects."""
    cfg = HymnPlusConfig(vocab_size=len(vocab), dim=32, n_layers=1, mlp_mult=2, seed=0)
    model = HymnPlus(cfg)

    ckpt = tmp_path / "tiny.npz"
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(ckpt, **sd)
    meta = {
        "arch": "hymn_plus_v1",
        "dim": 32,
        "n_layers": 1,
        "mlp_mult": 2,
        "vocab_size": len(vocab),
        "char_vocab": list(vocab),
        "seed": 0,
    }
    ckpt.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return ckpt


def test_from_checkpoint_round_trip(tmp_path):
    vocab = ["a", "b", "c", "d", "e"]
    ckpt = _save_tiny_checkpoint(tmp_path, vocab)
    sampler = HymnPlusSampler.from_checkpoint(ckpt, temperature=0.0)
    assert sampler.char_to_id["a"] == 0
    assert sampler.id_to_char[1] == "b"


def test_sampler_returns_str_of_requested_length(tmp_path):
    vocab = ["a", "b", "c", "d", "e", " "]
    ckpt = _save_tiny_checkpoint(tmp_path, vocab)
    sampler = HymnPlusSampler.from_checkpoint(ckpt, temperature=0.5, top_k=3)
    out = sampler("abc", n_tokens=10)
    assert isinstance(out, str)
    assert len(out) == 10
    # Every generated char must be in the vocab
    for c in out:
        assert c in vocab


def test_sampler_handles_unknown_prompt_chars(tmp_path):
    vocab = ["a", "b", "c"]
    ckpt = _save_tiny_checkpoint(tmp_path, vocab)
    sampler = HymnPlusSampler.from_checkpoint(ckpt, temperature=0.0)
    # Prompt has chars not in vocab; sampler should gracefully fall back
    out = sampler("XYZ", n_tokens=5)
    assert isinstance(out, str)
    assert len(out) == 5


def test_sampler_rejects_wrong_arch(tmp_path):
    ckpt = tmp_path / "bad.npz"
    np.savez_compressed(ckpt, dummy=np.zeros(1))
    ckpt.with_suffix(".json").write_text(json.dumps({"arch": "hymn_torch_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not 'hymn_plus_v1'"):
        HymnPlusSampler.from_checkpoint(ckpt)


def test_attaches_to_conscious_agent(tmp_path):
    """End-to-end: load a HYMN-Plus checkpoint, attach to ConsciousAgent,
    fire a KB-miss ask and confirm it returns a HYMN-sourced answer."""
    from rain.agent import ConsciousAgent

    vocab = list("abcdefghijklmnopqrstuvwxyz _")
    ckpt = _save_tiny_checkpoint(tmp_path, vocab)
    sampler = HymnPlusSampler.from_checkpoint(ckpt, temperature=0.0)

    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    agent.attach_hymn_sampler(sampler, n_tokens=20)
    ans = agent.ask("unknown_subject", "unknown_relation")
    # KB miss + HYMN attached -> epistemic='guess', source='hymn'
    assert ans.epistemic == "guess"
    assert ans.inference_source == "hymn"
    assert len(ans.text) > 0
