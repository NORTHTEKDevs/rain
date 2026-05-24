# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for HYMN-Plus."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_plus import HymnPlus, HymnPlusConfig, count_params  # noqa: E402


def _cfg(**overrides) -> HymnPlusConfig:
    base = {"vocab_size": 32, "dim": 64, "n_layers": 2, "mlp_mult": 2, "seed": 0}
    base.update(overrides)
    return HymnPlusConfig(**base)


def test_construction_and_param_count():
    m = HymnPlus(_cfg())
    # Embed: V*D = 32*64 = 2048
    # Per layer: recur 4*D*D + 1*D (W_g bias) = 4*64*64 + 64 = 16448
    #            mlp 3*D*(2*D) = 3*64*128 = 24576
    #            layernorms: 2 * 2*D = 256
    # 2 layers -> 2 * (16448 + 24576 + 256) = 82560
    # Final norm: 2*D = 128
    # Total = 2048 + 82560 + 128 = 84736 (tied weights)
    n = count_params(m)
    assert 50_000 < n < 200_000


def test_forward_shapes():
    m = HymnPlus(_cfg())
    tokens = torch.randint(0, 32, (3, 10))
    logits, states = m(tokens)
    assert logits.shape == (3, 10, 32)
    assert len(states) == m.config.n_layers
    for s in states:
        assert s.shape == (3, 64)


def test_chunked_forward_matches_full():
    """Stepping in chunks with state carry should match a single full forward."""
    m = HymnPlus(_cfg(seed=1))
    m.eval()
    tokens = torch.randint(0, 32, (1, 16))
    with torch.no_grad():
        full_logits, _ = m(tokens)
        states = None
        chunks = []
        for t in range(tokens.size(1)):
            l, states = m(tokens[:, t : t + 1], states=states)
            chunks.append(l)
        manual = torch.cat(chunks, dim=1)
    assert torch.allclose(full_logits, manual, atol=1e-4)


def test_smoke_train_step_reduces_loss():
    """Confirms gradients flow and the model can fit a tiny task."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = HymnPlus(_cfg(seed=0))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    tokens = torch.randint(0, 32, (4, 16))
    targets = torch.randint(0, 32, (4, 16))

    logits0, _ = m(tokens)
    loss0 = F.cross_entropy(logits0.reshape(-1, 32), targets.reshape(-1)).item()

    for _ in range(50):
        logits, _ = m(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    logits_f, _ = m(tokens)
    loss_f = F.cross_entropy(logits_f.reshape(-1, 32), targets.reshape(-1)).item()
    assert loss_f < loss0 * 0.7, f"insufficient learning: {loss0:.3f} -> {loss_f:.3f}"


def test_warm_start_codebook_propagates():
    rng = np.random.default_rng(42)
    cb = rng.choice([-1, 1], size=(32, 64)).astype(np.int16)
    m = HymnPlus(_cfg(init_codebook=cb))
    # embed.weight is the warm-start scaled by sqrt(D)
    expected = torch.as_tensor(cb, dtype=torch.float32) / (64**0.5)
    assert torch.allclose(m.embed.weight.data, expected, atol=1e-6)


def test_sample_returns_sequence():
    m = HymnPlus(_cfg())
    prompt = torch.randint(0, 32, (1, 5))
    out = m.sample(prompt, max_new=10, temperature=0.0)
    assert out.shape == (1, 15)
