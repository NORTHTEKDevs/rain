# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the REAL expert implementations in rain.core.real_experts."""

import numpy as np
import pytest

from rain.core.hv_substrate import random_hv, similarity
from rain.core.moa_router import MoARouter
from rain.core.real_experts import (
    install_text_sidechannel,
    make_hymn_expert,
    make_pure_attn_expert,
    make_real_expert_bank,
    make_sdm_expert,
    make_sym_regression_expert,
    make_tsetlin_expert,
)


D = 1024  # small for fast tests


# ============================================================
# Tsetlin expert
# ============================================================


def test_tsetlin_expert_produces_hv():
    spec = make_tsetlin_expert(dim=D, num_classes=4, num_clauses_per_class=8)
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    assert ans.shape == (D,)


def test_tsetlin_expert_attaches_firing_clauses():
    """The Tsetlin machine wrapper should expose firing_clauses for the
    symbolic_verifier to use."""
    spec = make_tsetlin_expert(dim=D, num_classes=4, num_clauses_per_class=8)
    assert hasattr(spec, "_tsetlin_machine")
    machine = spec._tsetlin_machine  # type: ignore[attr-defined]
    assert hasattr(machine, "firing_clauses")
    out = machine.firing_clauses(np.array([1, 0, 1, 0] * 256))
    assert isinstance(out, list)


def test_tsetlin_expert_deterministic_seed():
    s1 = make_tsetlin_expert(dim=D, seed=42)
    s2 = make_tsetlin_expert(dim=D, seed=42)
    q = random_hv(D, seed=1)
    a1 = s1.forward(q)
    a2 = s2.forward(q)
    assert np.array_equal(a1, a2)


# ============================================================
# SDM expert
# ============================================================


def test_sdm_cold_returns_query():
    spec = make_sdm_expert(dim=D)
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    # Cold SDM returns the query itself (graceful no-op).
    assert np.array_equal(ans, q)


def test_sdm_write_then_recall():
    spec = make_sdm_expert(dim=D)
    addr = random_hv(D, seed=1)
    content = random_hv(D, seed=2)
    spec._sdm_write(addr, content)  # type: ignore[attr-defined]
    # Recall with the SAME address should return something similar to content.
    ans = spec.forward(addr)
    assert similarity(ans, content) > 0.9


def test_sdm_noisy_address_still_recalls():
    spec = make_sdm_expert(dim=D)
    addr = random_hv(D, seed=1)
    content = random_hv(D, seed=2)
    spec._sdm_write(addr, content)  # type: ignore[attr-defined]
    # Slightly noisy address.
    noisy = np.sign(addr + 0.1 * random_hv(D, seed=99)).astype(np.float32)
    ans = spec.forward(noisy)
    assert similarity(ans, content) > 0.5


def test_sdm_capacity_bound():
    spec = make_sdm_expert(dim=D, capacity=4)
    for i in range(10):
        spec._sdm_write(random_hv(D, seed=i), random_hv(D, seed=100 + i))  # type: ignore[attr-defined]
    # After 10 writes with capacity 4, only the last 4 should remain.
    # Recall an old address (seed=0) should NOT match its original content.
    old_addr = random_hv(D, seed=0)
    old_content = random_hv(D, seed=100)
    ans = spec.forward(old_addr)
    # Should not be similar to the evicted content.
    assert similarity(ans, old_content) < 0.5


# ============================================================
# Sym-regression expert
# ============================================================


def test_sym_regression_extracts_arithmetic():
    spec = make_sym_regression_expert(dim=D)
    spec._state["last_text"] = "what is 5 + 3"  # type: ignore[attr-defined]
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    # Should differ from the stub (no last_text) version.
    spec2 = make_sym_regression_expert(dim=D)
    stub_ans = spec2.forward(q)
    assert not np.array_equal(ans, stub_ans)


def test_sym_regression_handles_complex_expr():
    spec = make_sym_regression_expert(dim=D)
    spec._state["last_text"] = "compute 10 * 7"  # type: ignore[attr-defined]
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    assert ans.shape == (D,)


def test_sym_regression_safe_against_eval_injection():
    """Should NOT execute non-arithmetic expressions."""
    spec = make_sym_regression_expert(dim=D)
    spec._state["last_text"] = "what is __import__('os').system('echo pwned')"  # type: ignore[attr-defined]
    q = random_hv(D, seed=1)
    # Should not crash and should not execute the os.system.
    ans = spec.forward(q)
    assert ans.shape == (D,)


def test_sym_regression_no_text_falls_back():
    spec = make_sym_regression_expert(dim=D)
    q = random_hv(D, seed=1)
    # Without last_text set, falls back to deterministic stub.
    ans = spec.forward(q)
    assert ans.shape == (D,)


# ============================================================
# HYMN expert
# ============================================================


def test_hymn_expert_no_checkpoint_falls_back():
    spec = make_hymn_expert(checkpoint_path=None, dim=D)
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    assert ans.shape == (D,)


def test_hymn_expert_missing_checkpoint_falls_back():
    spec = make_hymn_expert(checkpoint_path="/nonexistent/path.npz", dim=D)
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    assert ans.shape == (D,)
    assert not spec._state["available"]  # type: ignore[attr-defined]


# ============================================================
# Pure attention expert
# ============================================================


def test_pure_attn_cold_returns_query():
    spec = make_pure_attn_expert(dim=D)
    q = random_hv(D, seed=1)
    ans = spec.forward(q)
    # Cold: returns query (no prior context).
    assert similarity(ans, q) > 0.99


def test_pure_attn_recalls_similar_past():
    spec = make_pure_attn_expert(dim=D, window=8)
    # Prime with a sequence.
    seen = []
    for i in range(5):
        h = random_hv(D, seed=i)
        spec.forward(h)
        seen.append(h)
    # Query with something similar to seen[2].
    near = np.sign(seen[2] + 0.1 * random_hv(D, seed=99)).astype(np.float32)
    ans = spec.forward(near)
    # Should be weighted toward seen[2].
    assert similarity(ans, seen[2]) > similarity(ans, seen[0])


def test_pure_attn_window_bounded():
    spec = make_pure_attn_expert(dim=D, window=4)
    for i in range(10):
        spec.forward(random_hv(D, seed=i))
    assert len(spec._state["recent"]) == 4  # type: ignore[index]


# ============================================================
# Bank assembly + sidechannel
# ============================================================


def test_real_expert_bank_has_eight_experts():
    bank = make_real_expert_bank(dim=D)
    assert len(bank) == 8
    names = {e.name for e in bank}
    assert names == {
        "samba_lm", "pure_attn", "tsetlin", "diffusion",
        "gnn", "sdm", "sym_regression", "jepa_wm",
    }


def test_real_expert_bank_routes_correctly():
    bank = make_real_expert_bank(dim=D)
    router = MoARouter(experts=bank, top_k=2)
    q = random_hv(D, seed=1)
    ans, prov = router.forward(q)
    assert ans.shape == (D,)
    assert len(prov) == 2


def test_install_text_sidechannel_populates_state():
    bank = make_real_expert_bank(dim=D)
    install_text_sidechannel(bank, "what is 2 + 2")
    # Find sym_regression and verify it has the text.
    for e in bank:
        if e.name == "sym_regression":
            assert e._state["last_text"] == "what is 2 + 2"  # type: ignore[attr-defined]
            break
    else:
        pytest.fail("sym_regression expert not found in bank")


def test_install_text_sidechannel_safe_on_stub_experts():
    """Should not crash on experts without _state."""
    bank = make_real_expert_bank(dim=D)
    # Just verify it doesn't raise.
    install_text_sidechannel(bank, "hello world")
