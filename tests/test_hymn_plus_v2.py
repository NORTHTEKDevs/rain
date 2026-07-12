"""Tests for HYMN-Plus v2 (KB-attention architecture)."""

import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_plus_v2 import (  # noqa: E402
    HymnPlusV2,
    HymnPlusV2Config,
    KbAttention,
    count_params,
)


def _cfg(**overrides) -> HymnPlusV2Config:
    base = {
        "vocab_size": 64,
        "dim": 64,
        "n_layers": 2,
        "mlp_mult": 2,
        "kb_size": 32,
        "kb_top_k": 4,
        "seed": 0,
    }
    base.update(overrides)
    return HymnPlusV2Config(**base)


def test_kb_attention_shapes():
    kb = KbAttention(dim=64, kb_size=32, top_k=4)
    x = torch.randn(3, 8, 64)  # (B, T, D)
    out = kb(x)
    assert out.shape == (3, 8, 64)


def test_kb_attention_returns_weights_when_asked():
    kb = KbAttention(dim=64, kb_size=32, top_k=4)
    x = torch.randn(2, 5, 64)
    out, attn = kb(x, return_attn=True)
    assert out.shape == (2, 5, 64)
    assert attn.shape == (2, 5, 32)
    # Softmax: each row sums to 1
    sums = attn.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_kb_attention_top_k_zeros_non_top_facts():
    """Only the top-K weights should be nonzero per (batch, time) position."""
    kb = KbAttention(dim=32, kb_size=16, top_k=3)
    x = torch.randn(1, 1, 32)
    _, attn = kb(x, return_attn=True)
    # 16 facts but only 3 should have nonzero weight
    nonzero = (attn[0, 0] > 1e-6).sum().item()
    assert nonzero == 3


def test_set_kb_updates_buffer():
    kb = KbAttention(dim=32, kb_size=8, top_k=4)
    new_kb = torch.ones(8, 32) * 0.5
    kb.set_kb(new_kb)
    torch.testing.assert_close(kb.kb, new_kb)


def test_v2_forward_shapes():
    m = HymnPlusV2(_cfg())
    tokens = torch.randint(0, 64, (3, 10))
    logits, states = m(tokens)
    assert logits.shape == (3, 10, 64)
    assert len(states) == m.config.n_layers


def test_v2_chunked_matches_full():
    """Stepping in chunks must match a single full forward (state continuity)."""
    m = HymnPlusV2(_cfg(seed=1))
    m.eval()
    tokens = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        full, _ = m(tokens)
        states = None
        chunks = []
        for t in range(tokens.size(1)):
            l, states = m(tokens[:, t : t + 1], states=states)
            chunks.append(l)
        manual = torch.cat(chunks, dim=1)
    assert torch.allclose(full, manual, atol=1e-4)


def test_v2_return_kb_attn_traces_each_kb_layer():
    m = HymnPlusV2(_cfg())
    tokens = torch.randint(0, 64, (1, 5))
    _, _, kb_attns = m(tokens, return_kb_attn=True)
    # All layers have KB-attn enabled by default -> all entries non-None
    assert all(a is not None for a in kb_attns)
    for a in kb_attns:
        assert a.shape == (1, 5, 32)


def test_v2_smoke_train_step_reduces_loss():
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = HymnPlusV2(_cfg(seed=0))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    tokens = torch.randint(0, 64, (4, 16))
    targets = torch.randint(0, 64, (4, 16))
    logits0, _ = m(tokens)
    loss0 = F.cross_entropy(logits0.reshape(-1, 64), targets.reshape(-1)).item()
    for _ in range(40):
        logits, _ = m(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    logits_f, _ = m(tokens)
    loss_f = F.cross_entropy(logits_f.reshape(-1, 64), targets.reshape(-1)).item()
    assert loss_f < loss0 * 0.7


def test_v2_selective_kb_attn_in_layers():
    """Only blocks listed in kb_attn_in_layers should have use_kb_attn=True."""
    cfg = _cfg(n_layers=4, kb_attn_in_layers=(1, 3))
    m = HymnPlusV2(cfg)
    flags = [b.use_kb_attn for b in m.blocks]
    assert flags == [False, True, False, True]


def test_v2_set_kb_writes_to_all_kb_blocks():
    cfg = _cfg(n_layers=3, kb_attn_in_layers=(0, 2))
    m = HymnPlusV2(cfg)
    new_kb = torch.ones(cfg.kb_size, cfg.dim) * 0.25
    m.set_kb(new_kb)
    # Only blocks 0 and 2 have KB-attn
    torch.testing.assert_close(m.blocks[0].kb_attn.kb, new_kb)
    torch.testing.assert_close(m.blocks[2].kb_attn.kb, new_kb)


def test_v2_kb_facts_influence_output():
    """Verify the model actually USES the KB: setting a different KB should
    change the output for the same input."""
    cfg = _cfg(seed=42)
    m = HymnPlusV2(cfg)
    # Force the KB-attention output projection to non-zero so KB actually matters.
    with torch.no_grad():
        for block in m.blocks:
            if block.use_kb_attn:
                torch.nn.init.normal_(block.kb_attn.W_o.weight, std=0.5)
    m.eval()
    tokens = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        out_a, _ = m(tokens)
        # Swap the KB
        kb_b = torch.randn(cfg.kb_size, cfg.dim) * 2.0
        m.set_kb(kb_b)
        out_b, _ = m(tokens)
    # Logits should differ when the KB changes
    diff = (out_a - out_b).abs().mean().item()
    assert diff > 1e-4, f"KB swap had no measurable effect on logits (diff={diff:.6f})"


def test_v2_sample_returns_sequence():
    m = HymnPlusV2(_cfg())
    prompt = torch.randint(0, 64, (1, 4))
    out = m.sample(prompt, max_new=8, temperature=0.0)
    assert out.shape == (1, 12)


def test_v2_param_count_with_kb_attn_higher_than_without():
    m_with = HymnPlusV2(_cfg(n_layers=2))
    cfg_without = _cfg(n_layers=2, kb_attn_in_layers=())
    m_without = HymnPlusV2(cfg_without)
    # KB-attn adds 4 DxD matrices per block
    assert count_params(m_with) > count_params(m_without)
