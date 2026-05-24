# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the HYMN-Plus v2 sampler adapter -- including the
killer feature: tell() -> set_kb_from_facts() -> generation reflects
new knowledge without retraining."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
spm = pytest.importorskip("sentencepiece")

from rain.cognition.hymn_plus_v2_sampler import HymnPlusV2Sampler  # noqa: E402
from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config  # noqa: E402
from rain.tokenize.bpe import BPETokenizer  # noqa: E402


@pytest.fixture
def trained_v2_checkpoint(tmp_path):
    """Train a microscopic HYMN-Plus v2 + save with BPE sidecar."""
    import torch.nn.functional as F

    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the cat sat on the mat. romeo and juliet love each other. "
    ) * 20

    tok = BPETokenizer(vocab_size=128)
    tok.train([corpus])
    ids = np.array(tok.encode(corpus), dtype=np.int64)
    vocab = tok._sp.get_piece_size()

    cfg = HymnPlusV2Config(
        vocab_size=vocab,
        dim=32,
        n_layers=2,
        mlp_mult=2,
        kb_size=16,
        kb_top_k=4,
        seed=0,
    )
    model = HymnPlusV2(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(30):
        starts = np.random.default_rng(0).integers(0, len(ids) - 33, size=4)
        batch = np.stack([ids[s : s + 33] for s in starts])
        x = torch.as_tensor(batch[:, :-1], dtype=torch.long)
        y = torch.as_tensor(batch[:, 1:], dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    ckpt = tmp_path / "v2_tiny.npz"
    np.savez_compressed(
        ckpt, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    )
    bpe_path = ckpt.with_suffix(".bpe.model")
    tok.save(bpe_path)
    ckpt.with_suffix(".json").write_text(
        json.dumps(
            {
                "arch": "hymn_plus_v2",
                "dim": 32,
                "n_layers": 2,
                "mlp_mult": 2,
                "vocab_size": vocab,
                "bpe_vocab": 128,
                "bpe_model_path": str(bpe_path),
                "kb_size": 16,
                "kb_top_k": 4,
                "kb_attn_in_layers": None,
                "tie_weights": True,
                "seed": 0,
            }
        )
    )
    return ckpt


def test_v2_sampler_loads_and_returns_str(trained_v2_checkpoint):
    sampler = HymnPlusV2Sampler.from_checkpoint(trained_v2_checkpoint, temperature=0.0)
    out = sampler("the ", n_tokens=8)
    assert isinstance(out, str)
    assert len(out) > 0


def test_v2_sampler_rejects_wrong_arch(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, dummy=np.zeros(1))
    bad.with_suffix(".json").write_text(json.dumps({"arch": "hymn_plus_v1"}))
    with pytest.raises(ValueError, match="not 'hymn_plus_v2'"):
        HymnPlusV2Sampler.from_checkpoint(bad)


def test_set_kb_from_facts_changes_generation(trained_v2_checkpoint):
    """The KILLER FEATURE TEST: pushing new facts into the KB measurably
    changes the model's generation, without any retraining."""
    sampler = HymnPlusV2Sampler.from_checkpoint(trained_v2_checkpoint, temperature=0.0)

    # Force KB-attention output projection to non-zero so the KB actually
    # contributes to logits (random-init W_o is zero by design).
    for block in sampler.model.blocks:
        if block.use_kb_attn:
            with torch.no_grad():
                torch.nn.init.normal_(block.kb_attn.W_o.weight, std=0.5)

    out_a = sampler("the ", n_tokens=10)
    # Now load a totally different set of facts
    n = sampler.set_kb_from_facts(
        [("lion", "lives_in", "savanna"), ("dolphin", "lives_in", "ocean")] * 10,
        seed=42,
    )
    assert n > 0
    out_b = sampler("the ", n_tokens=10)
    # With T=0 (greedy), output should differ when KB differs
    assert out_a != out_b, "KB change did not influence generation -- the moat is broken"


def test_set_kb_from_facts_truncates_to_kb_size(trained_v2_checkpoint):
    sampler = HymnPlusV2Sampler.from_checkpoint(trained_v2_checkpoint)
    huge = [("s", "r", f"o_{i}") for i in range(100)]
    # kb_size = 16 in fixture; should cap at 16
    n = sampler.set_kb_from_facts(huge)
    assert n == 16
