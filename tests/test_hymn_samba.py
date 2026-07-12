"""Tests for HYMN-Samba (Mamba S6 + Sliding-Window-Attention hybrid)."""

import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_samba import (  # noqa: E402
    HymnSamba,
    HymnSambaConfig,
    MambaLayer,
    SlidingWindowAttention,
    SwaLayer,
)


def _cfg(**overrides) -> HymnSambaConfig:
    base = {
        "vocab_size": 32,
        "dim": 32,
        "n_layers": 4,
        "mlp_mult": 2,
        "n_heads": 2,
        "max_seq_len": 64,
        "window_size": 16,
        "dropout": 0.0,
        "d_state": 8,
        "expand": 2,
        "seed": 0,
    }
    base.update(overrides)
    return HymnSambaConfig(**base)


def test_swa_causal_and_window_respected():
    """SWA: token at position i should only attend to tokens in
    [max(0, i-window+1), i]. Verify by changing future tokens and
    confirming positions before the window stay unchanged."""
    attn = SlidingWindowAttention(dim=32, n_heads=4, window_size=8)
    attn.eval()
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        y_a = attn(x)
        # Change a future token (position 15) -- should not affect
        # outputs at positions <= 14 (causal) AND should not affect
        # positions where 15 is OUTSIDE their window (positions <= 15-8 = 7)
        x2 = x.clone()
        x2[0, 15] = torch.randn(32)
        y_b = attn(x2)
    # Causal: positions 0..14 must not see position 15
    torch.testing.assert_close(y_a[:, :15], y_b[:, :15], atol=1e-4, rtol=1e-4)


def test_swa_window_excludes_distant_past():
    """SWA: token at position i must NOT attend to position j < i - window + 1.
    Verify by changing a distant past token and confirming token i is unchanged."""
    attn = SlidingWindowAttention(dim=32, n_heads=4, window_size=4)
    attn.eval()
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        y_a = attn(x)
        # Change position 5; positions 9+ should not see it (window=4 means
        # position 9 attends to [6, 7, 8, 9], position 5 is outside)
        x2 = x.clone()
        x2[0, 5] = torch.randn(32)
        y_b = attn(x2)
    # Position 9 should be unchanged
    torch.testing.assert_close(y_a[:, 9], y_b[:, 9], atol=1e-4, rtol=1e-4)


def test_mamba_layer_is_causal():
    """MambaLayer: changing position T must not change outputs at positions < T."""
    cfg = _cfg()
    layer = MambaLayer(cfg)
    layer.eval()
    x = torch.randn(1, 16, cfg.dim)
    with torch.no_grad():
        y_a = layer(x)
        x2 = x.clone()
        x2[0, -1] = torch.randn(cfg.dim)
        y_b = layer(x2)
    torch.testing.assert_close(y_a[:, :-1], y_b[:, :-1], atol=1e-4, rtol=1e-4)


def test_swa_layer_is_causal():
    """SwaLayer: same causality check."""
    cfg = _cfg()
    layer = SwaLayer(cfg)
    layer.eval()
    x = torch.randn(1, 12, cfg.dim)
    with torch.no_grad():
        y_a = layer(x)
        x2 = x.clone()
        x2[0, -1] = torch.randn(cfg.dim)
        y_b = layer(x2)
    torch.testing.assert_close(y_a[:, :-1], y_b[:, :-1], atol=1e-4, rtol=1e-4)


def test_full_model_forward_shape():
    cfg = _cfg(vocab_size=64, dim=64, n_layers=4)
    m = HymnSamba(cfg)
    tokens = torch.randint(0, 64, (3, 16))
    logits = m(tokens)
    assert logits.shape == (3, 16, 64)


def test_full_model_is_causal():
    cfg = _cfg(seed=1)
    m = HymnSamba(cfg)
    m.eval()
    tokens = torch.randint(0, 32, (1, 16))
    with torch.no_grad():
        l_a = m(tokens)
        tokens2 = tokens.clone()
        tokens2[0, -1] = (tokens2[0, -1] + 7) % 32
        l_b = m(tokens2)
    torch.testing.assert_close(l_a[:, :-1], l_b[:, :-1], atol=1e-4, rtol=1e-4)


def test_layer_pattern_ms_alternates():
    """layer_pattern='MS' with n_layers=4 should yield M, S, M, S."""
    cfg = _cfg(n_layers=4, layer_pattern="MS")
    m = HymnSamba(cfg)
    kinds = [type(b).__name__ for b in m.blocks]
    assert kinds == ["MambaLayer", "SwaLayer", "MambaLayer", "SwaLayer"]


def test_layer_pattern_mms_pattern():
    cfg = _cfg(n_layers=6, layer_pattern="MMS")
    m = HymnSamba(cfg)
    kinds = [type(b).__name__ for b in m.blocks]
    assert kinds == [
        "MambaLayer",
        "MambaLayer",
        "SwaLayer",
        "MambaLayer",
        "MambaLayer",
        "SwaLayer",
    ]


def test_invalid_layer_pattern_raises():
    with pytest.raises(ValueError, match="layer_pattern"):
        HymnSamba(_cfg(layer_pattern="XYZ"))


def test_smoke_train_reduces_loss():
    import torch.nn.functional as F

    torch.manual_seed(0)
    cfg = _cfg(seed=0, n_layers=2)
    m = HymnSamba(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    tokens = torch.randint(0, 32, (4, 16))
    targets = torch.randint(0, 32, (4, 16))
    l0 = m(tokens)
    loss0 = F.cross_entropy(l0.reshape(-1, 32), targets.reshape(-1)).item()
    for _ in range(40):
        logits = m(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    l_f = m(tokens)
    loss_f = F.cross_entropy(l_f.reshape(-1, 32), targets.reshape(-1)).item()
    assert loss_f < loss0 * 0.7, f"insufficient learning: {loss0:.3f} -> {loss_f:.3f}"


def test_sample_produces_extended_sequence():
    cfg = _cfg(vocab_size=16, dim=32, n_layers=2)
    m = HymnSamba(cfg)
    prompt = torch.randint(0, 16, (1, 5))
    out = m.sample(prompt, max_new=8, temperature=0.0)
    assert out.shape == (1, 13)
