"""Tests for HYMN-Pro (attention-based LM core)."""

import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_pro import (  # noqa: E402
    HymnPro,
    HymnProBlock,
    HymnProConfig,
    MultiHeadCausalAttention,
    SwiGLU,
    count_params,
)


def _cfg(**overrides) -> HymnProConfig:
    base = {
        "vocab_size": 32,
        "dim": 32,
        "n_layers": 2,
        "n_heads": 2,
        "mlp_mult": 2,
        "max_seq_len": 64,
        "dropout": 0.0,
        "seed": 0,
    }
    base.update(overrides)
    return HymnProConfig(**base)


def test_attention_shapes():
    attn = MultiHeadCausalAttention(dim=32, n_heads=4)
    x = torch.randn(2, 16, 32)
    out = attn(x)
    assert out.shape == (2, 16, 32)


def test_attention_is_causal():
    """A token's representation must not depend on FUTURE tokens.
    Smoke this by changing the LAST token and confirming earlier tokens
    are unchanged."""
    attn = MultiHeadCausalAttention(dim=32, n_heads=4)
    attn.eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        out_a = attn(x)
        # Change the LAST token; first 9 outputs should be IDENTICAL
        x2 = x.clone()
        x2[:, -1] = torch.randn(32)
        out_b = attn(x2)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


def test_swiglu_shape():
    mlp = SwiGLU(dim=32, mult=4)
    x = torch.randn(3, 8, 32)
    assert mlp(x).shape == (3, 8, 32)


def test_block_shape():
    block = HymnProBlock(_cfg())
    x = torch.randn(2, 16, 32)
    assert block(x).shape == (2, 16, 32)


def test_forward_shape_and_param_count():
    cfg = _cfg(dim=64, n_layers=4, n_heads=4, vocab_size=128)
    m = HymnPro(cfg)
    tokens = torch.randint(0, 128, (3, 24))
    logits = m(tokens)
    assert logits.shape == (3, 24, 128)
    # 4 layers x (~4*D^2 attn + 12*D^2 mlp) + V*D embed + pos*D
    # = 4 * 16 * D^2 + 128*64 + 64*64 = 64*64*64 + 8192 + 4096 = ~270K
    # Just sanity check it's in the right ballpark
    n = count_params(m)
    assert 100_000 < n < 500_000


def test_full_forward_is_causal():
    """The full model must be causal: changing the LAST token in the
    input should not change predictions for any earlier position."""
    cfg = _cfg(seed=1)
    m = HymnPro(cfg)
    m.eval()
    tokens = torch.randint(0, 32, (1, 12))
    with torch.no_grad():
        l_a = m(tokens)
        tokens2 = tokens.clone()
        tokens2[0, -1] = (tokens2[0, -1] + 7) % 32
        l_b = m(tokens2)
    # All but the last logit position must be identical
    assert torch.allclose(l_a[:, :-1], l_b[:, :-1], atol=1e-5)


def test_smoke_train_step_reduces_loss():
    """One gradient run on a tiny task must reduce loss meaningfully.
    Confirms gradient flow through attention + MLP + tied head."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    cfg = _cfg(seed=0)
    m = HymnPro(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    tokens = torch.randint(0, 32, (4, 16))
    targets = torch.randint(0, 32, (4, 16))
    l0 = m(tokens)
    loss0 = F.cross_entropy(l0.reshape(-1, 32), targets.reshape(-1)).item()
    for _ in range(50):
        logits = m(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    l_f = m(tokens)
    loss_f = F.cross_entropy(l_f.reshape(-1, 32), targets.reshape(-1)).item()
    assert loss_f < loss0 * 0.6, f"loss did not drop enough: {loss0:.3f} -> {loss_f:.3f}"


def test_sample_returns_extended_sequence():
    m = HymnPro(_cfg())
    prompt = torch.randint(0, 32, (1, 5))
    out = m.sample(prompt, max_new=10, temperature=0.0)
    assert out.shape == (1, 15)


def test_sample_truncates_context_to_max_seq_len():
    """Long prompts must still work (model truncates context)."""
    cfg = _cfg(max_seq_len=32)
    m = HymnPro(cfg)
    long_prompt = torch.randint(0, 32, (1, 64))
    out = m.sample(long_prompt, max_new=5, temperature=0.0)
    assert out.shape == (1, 69)


def test_tied_weights_have_lm_head_none():
    m_tied = HymnPro(_cfg(tie_weights=True))
    m_untied = HymnPro(_cfg(tie_weights=False))
    assert m_tied.lm_head is None
    assert m_untied.lm_head is not None
    # Untied should have MORE parameters (extra V*D matrix)
    assert count_params(m_untied) > count_params(m_tied)
