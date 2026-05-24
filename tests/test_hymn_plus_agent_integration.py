# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""End-to-end integration test: real HYMN-Plus checkpoint -> ConsciousAgent ->
ask flow with KB hit + KB miss paths.

Doesn't require a pretrained checkpoint from disk -- we train a microscopic
one in the fixture (1 layer, dim=32, 5 steps) so the test is self-contained
and CI-runnable.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rain.agent import ConsciousAgent  # noqa: E402
from rain.cognition.hymn_plus_sampler import HymnPlusSampler  # noqa: E402
from rain.core.hymn_plus import HymnPlus, HymnPlusConfig  # noqa: E402


@pytest.fixture
def trained_checkpoint(tmp_path):
    """Train a microscopic HYMN-Plus on a few hundred chars + save it."""
    import torch.nn.functional as F

    vocab = list("abcdefghijklmnopqrstuvwxyz _.,")
    char_to_id = {c: i for i, c in enumerate(vocab)}
    text = "the cat sat on the mat. the dog ran fast. " * 20
    ids = np.array([char_to_id[c] for c in text if c in char_to_id], dtype=np.int64)

    cfg = HymnPlusConfig(vocab_size=len(vocab), dim=32, n_layers=1, mlp_mult=2, seed=0)
    model = HymnPlus(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(20):
        starts = np.random.default_rng(0).integers(0, len(ids) - 33, size=4)
        batch = np.stack([ids[s : s + 33] for s in starts])
        x = torch.as_tensor(batch[:, :-1], dtype=torch.long)
        y = torch.as_tensor(batch[:, 1:], dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    ckpt = tmp_path / "tiny_hymn_plus.npz"
    np.savez_compressed(
        ckpt, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    )
    ckpt.with_suffix(".json").write_text(
        json.dumps(
            {
                "arch": "hymn_plus_v1",
                "dim": 32,
                "n_layers": 1,
                "mlp_mult": 2,
                "vocab_size": len(vocab),
                "char_vocab": vocab,
                "seed": 0,
            }
        )
    )
    return ckpt


def test_kb_hit_does_not_trigger_hymn(trained_checkpoint):
    """When the KB has a fact, the agent should answer from the KB and NOT
    invoke the HYMN sampler."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    sampler = HymnPlusSampler.from_checkpoint(trained_checkpoint, temperature=0.0)
    agent.attach_hymn_sampler(sampler, n_tokens=30)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    # KB hit: source is structural, not 'hymn'
    assert ans.inference_source == "direct"
    assert ans.epistemic in ("know", "think")
    assert "savanna" in ans.text


def test_kb_miss_triggers_hymn_with_guess_epistemic(trained_checkpoint):
    """When the KB doesn't know, the agent uses HYMN-Plus and tags the
    answer as guess/hymn."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    sampler = HymnPlusSampler.from_checkpoint(trained_checkpoint, temperature=0.0)
    agent.attach_hymn_sampler(sampler, n_tokens=20)
    ans = agent.ask("unknown_subject", "unknown_relation")
    assert ans.inference_source == "hymn"
    assert ans.epistemic == "guess"
    assert len(ans.text) > 0


def test_kb_then_tell_then_ask_round_trip(trained_checkpoint):
    """tell() learns a new fact, ask() retrieves it -- HYMN stays out of
    the loop because the KB now has the answer."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    sampler = HymnPlusSampler.from_checkpoint(trained_checkpoint, temperature=0.0)
    agent.attach_hymn_sampler(sampler, n_tokens=20)
    # First ask: KB miss -> HYMN fallback
    miss = agent.ask("tiger", "lives_in")
    assert miss.inference_source == "hymn"
    # Now teach it
    agent.tell("tiger", "lives_in", "jungle")
    # Second ask: KB hit
    hit = agent.ask("tiger", "lives_in")
    assert hit.inference_source == "direct"
    assert "jungle" in hit.text


def test_hymn_plus_sampler_is_deterministic_at_zero_temperature(trained_checkpoint):
    """At temperature=0, sampling is greedy argmax -- same prompt should
    give same output across calls."""
    sampler = HymnPlusSampler.from_checkpoint(trained_checkpoint, temperature=0.0)
    out1 = sampler("the ", n_tokens=20)
    out2 = sampler("the ", n_tokens=20)
    assert out1 == out2
