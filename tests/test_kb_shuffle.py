"""Tests for the v3 KB-shuffle training primitive."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config  # noqa: E402
from scripts.pretrain_hymn_plus_v2 import _kb_shuffle  # noqa: E402


def _make_model(kb_size: int = 32, dim: int = 16, n_layers: int = 2) -> HymnPlusV2:
    cfg = HymnPlusV2Config(
        vocab_size=32,
        dim=dim,
        n_layers=n_layers,
        mlp_mult=2,
        kb_size=kb_size,
        kb_top_k=4,
        seed=0,
    )
    return HymnPlusV2(cfg)


def test_kb_shuffle_replaces_expected_fraction():
    """frac=0.25 with 32 KB entries should replace exactly 8 rows."""
    m = _make_model(kb_size=32)
    kb_before = m.blocks[0].kb_attn.kb.clone()
    rng = np.random.default_rng(0)
    _kb_shuffle(m, frac=0.25, rng=rng, device=torch.device("cpu"))
    kb_after = m.blocks[0].kb_attn.kb
    # Per-row equality: count rows that are unchanged
    same = (kb_before == kb_after).all(dim=-1).sum().item()
    # 32 total, 8 replaced -> 24 unchanged
    assert same == 24


def test_kb_shuffle_frac_zero_is_noop():
    m = _make_model()
    kb_before = m.blocks[0].kb_attn.kb.clone()
    rng = np.random.default_rng(0)
    _kb_shuffle(m, frac=0.0, rng=rng, device=torch.device("cpu"))
    assert torch.equal(kb_before, m.blocks[0].kb_attn.kb)


def test_kb_shuffle_affects_all_kb_blocks():
    m = _make_model(n_layers=3)
    snapshots = [b.kb_attn.kb.clone() for b in m.blocks if b.use_kb_attn]
    rng = np.random.default_rng(0)
    _kb_shuffle(m, frac=0.5, rng=rng, device=torch.device("cpu"))
    for snap, block in zip(snapshots, [b for b in m.blocks if b.use_kb_attn], strict=True):
        # At least one row in each block should differ
        diff = (snap != block.kb_attn.kb).any(dim=-1).sum().item()
        assert diff > 0


def test_kb_shuffle_new_rows_are_bipolar():
    """Replaced rows must contain only +/-1 values (bipolar)."""
    m = _make_model(kb_size=16, dim=8)
    rng = np.random.default_rng(0)
    _kb_shuffle(m, frac=1.0, rng=rng, device=torch.device("cpu"))
    kb = m.blocks[0].kb_attn.kb
    unique_vals = torch.unique(kb).tolist()
    assert set(unique_vals).issubset({-1.0, 1.0})


def test_kb_shuffle_is_deterministic_with_same_rng():
    """Same rng seed -> same KB after shuffle."""
    m1 = _make_model(kb_size=16)
    m2 = _make_model(kb_size=16)
    # Same starting KB
    m2.blocks[0].kb_attn.kb.copy_(m1.blocks[0].kb_attn.kb)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    _kb_shuffle(m1, frac=0.5, rng=rng1, device=torch.device("cpu"))
    _kb_shuffle(m2, frac=0.5, rng=rng2, device=torch.device("cpu"))
    assert torch.equal(m1.blocks[0].kb_attn.kb, m2.blocks[0].kb_attn.kb)
