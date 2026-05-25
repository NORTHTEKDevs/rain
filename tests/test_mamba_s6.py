# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the proper Mamba S6 block + HYMN-Mamba assembly."""

import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_mamba import HymnMamba, HymnMambaConfig, count_params  # noqa: E402
from rain.core.mamba_s6 import MambaS6Block, MambaS6Config  # noqa: E402


def test_s6_block_shape():
    cfg = MambaS6Config(dim=32, d_state=8, d_conv=4, expand=2)
    block = MambaS6Block(cfg)
    x = torch.randn(2, 10, 32)
    assert block(x).shape == (2, 10, 32)


def test_s6_block_is_causal():
    """Mamba is a sequence model; changing token T must not change tokens < T."""
    cfg = MambaS6Config(dim=32, d_state=8, d_conv=4, expand=2)
    block = MambaS6Block(cfg)
    block.eval()
    x = torch.randn(1, 12, 32)
    with torch.no_grad():
        y_a = block(x)
        x2 = x.clone()
        x2[0, -1] = torch.randn(32)
        y_b = block(x2)
    # All but the last position should match
    torch.testing.assert_close(y_a[:, :-1], y_b[:, :-1], atol=1e-4, rtol=1e-4)


def test_s6_block_finite_under_long_seq():
    cfg = MambaS6Config(dim=16, d_state=4, d_conv=4, expand=2)
    block = MambaS6Block(cfg)
    x = torch.randn(1, 200, 16)
    with torch.no_grad():
        y = block(x)
    assert torch.isfinite(y).all()


def test_hymn_mamba_forward_shape():
    cfg = HymnMambaConfig(vocab_size=32, dim=64, n_layers=2, mlp_mult=2, d_state=8)
    m = HymnMamba(cfg)
    tokens = torch.randint(0, 32, (3, 16))
    logits = m(tokens)
    assert logits.shape == (3, 16, 32)


def test_hymn_mamba_param_count_reasonable():
    cfg = HymnMambaConfig(vocab_size=64, dim=128, n_layers=4, mlp_mult=4, d_state=16, expand=2)
    m = HymnMamba(cfg)
    n = count_params(m)
    # Embed: 64*128 = 8K. Per layer (d_inner=256):
    #   in_proj: 128 * 512 = 65K
    #   conv: 256 * 4 = 1K (per-channel grouped)
    #   x_proj: 256 * (dt_rank + 2*16); dt_rank = ceil(128/16) = 8 -> 256*40 = 10K
    #   dt_proj: 8 * 256 = 2K + 256 bias
    #   A_log: 256 * 16 = 4K; D: 256
    #   out_proj: 256 * 128 = 32K
    #   mlp (SwiGLU, 4x): 3 * 128 * 512 = 196K
    # 4 layers -> ~1.2M total
    assert 500_000 < n < 5_000_000


def test_smoke_train_step_reduces_loss():
    import torch.nn.functional as F

    torch.manual_seed(0)
    cfg = HymnMambaConfig(vocab_size=32, dim=32, n_layers=1, mlp_mult=2, d_state=8)
    m = HymnMamba(cfg)
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


def test_hymn_mamba_sample_returns_extended_sequence():
    cfg = HymnMambaConfig(vocab_size=16, dim=32, n_layers=1, mlp_mult=2, d_state=4)
    m = HymnMamba(cfg)
    prompt = torch.randint(0, 16, (1, 5))
    out = m.sample(prompt, max_new=8, temperature=0.0)
    assert out.shape == (1, 13)
