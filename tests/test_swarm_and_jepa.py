# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for multi-agent VSA swarm + JEPA world-model expert."""

import numpy as np
import pytest

from rain.core.hv_substrate import hash_to_hv, random_hv, similarity
from rain.core.jepa_expert import (
    JEPAState,
    make_jepa_expert,
    train_jepa_from_episodic,
)
from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.swarm import RainNetSwarm, SwarmMember


D = 1024


# ============================================================
# Multi-agent swarm
# ============================================================


def test_swarm_empty_initially():
    s = RainNetSwarm(dim=D)
    assert len(s) == 0
    result = s.answer("anything")
    assert "empty" in result.text


def test_swarm_add_member():
    s = RainNetSwarm(dim=D)
    m = s.add_member("test", domain_text="testing domain words")
    assert m.name == "test"
    assert m.domain_hv.shape == (D,)
    assert len(s) == 1


def test_swarm_duplicate_member_raises():
    s = RainNetSwarm(dim=D)
    s.add_member("a", domain_text="x")
    with pytest.raises(ValueError, match="already exists"):
        s.add_member("a", domain_text="y")


def test_swarm_route_selects_top_k():
    s = RainNetSwarm(dim=D)
    s.add_member("aviation", domain_text="aircraft FAA airworthiness Cessna Piper")
    s.add_member("legal", domain_text="contract law clause attorney")
    s.add_member("medical", domain_text="patient doctor diagnosis treatment")
    # An aviation-style query should route to the aviation member.
    query_hv = s._shared_encoder.encode("text", "What FAA AD applies to my Cessna")
    routing = s.route(query_hv, top_k=2)
    assert len(routing.members) == 2
    names = {n for n, _w in routing.members}
    assert "aviation" in names


def test_swarm_answer_aggregates():
    s = RainNetSwarm(dim=D)
    s.add_member("aviation", domain_text="aircraft FAA airworthiness")
    s.add_member("legal", domain_text="contract law clause")
    s["aviation"].net.ingest_fact(
        "AD 2024-01-05 applies to all Cessna 172 R/S models for fuel system inspection.",
        fact_id="ad1",
    )
    s["legal"].net.ingest_fact("A contract requires offer, acceptance, and consideration.", fact_id="law1")
    result = s.answer("What AD applies to my Cessna?", top_k=2)
    assert result.text is not None
    assert len(result.contributing_members) >= 1


def test_swarm_remove_member():
    s = RainNetSwarm(dim=D)
    s.add_member("a", domain_text="x")
    s.add_member("b", domain_text="y")
    assert s.remove_member("a") is True
    assert len(s) == 1
    assert s.remove_member("nonexistent") is False


def test_swarm_getitem():
    s = RainNetSwarm(dim=D)
    s.add_member("test", domain_text="x")
    m = s["test"]
    assert m.name == "test"
    with pytest.raises(KeyError):
        _ = s["nonexistent"]


def test_swarm_stats():
    s = RainNetSwarm(dim=D)
    s.add_member("a", domain_text="x")
    s["a"].net.ingest_fact("a fact")
    s.answer("test query")
    stats = s.stats()
    assert stats["n_members"] == 1
    assert stats["members"][0]["kb_size"] == 1
    assert stats["members"][0]["invocations"] >= 1


def test_swarm_member_invocation_tracking():
    s = RainNetSwarm(dim=D)
    s.add_member("aviation", domain_text="aircraft FAA")
    s.add_member("legal", domain_text="contract law clause")
    # Aviation query should bump aviation invocations more.
    for _ in range(3):
        s.answer("What FAA AD applies", top_k=1)
    av = s["aviation"]
    le = s["legal"]
    assert av.invocations >= le.invocations


# ============================================================
# JEPA world-model expert
# ============================================================


def test_jepa_state_predict_returns_hv():
    state = JEPAState(dim=D, n_permutations=4)
    q = random_hv(D, seed=1)
    pred = state.predict(q)
    assert pred.shape == (D,)


def test_jepa_train_changes_prediction():
    state = JEPAState(dim=D, n_permutations=4)
    q = random_hv(D, seed=1)
    target = random_hv(D, seed=2)
    pred_before = state.predict(q).copy()
    for _ in range(20):
        state.train_step(q, target)
    pred_after = state.predict(q)
    # Trained prediction should be MORE similar to target than untrained.
    sim_before = similarity(pred_before, target)
    sim_after = similarity(pred_after, target)
    assert sim_after > sim_before


def test_jepa_expert_spec():
    spec = make_jepa_expert(dim=D)
    assert spec.name == "jepa_wm"
    assert hasattr(spec, "_jepa_state")
    q = random_hv(D, seed=1)
    out = spec.forward(q)
    assert out.shape == (D,)


def test_jepa_expert_in_real_bank():
    """JEPA should be wired as a real expert in make_real_expert_bank now."""
    from rain.core.real_experts import make_real_expert_bank

    bank = make_real_expert_bank(dim=D)
    jepa = next((e for e in bank if e.name == "jepa_wm"), None)
    assert jepa is not None
    # Has the state (real impl), not just a stub.
    assert hasattr(jepa, "_jepa_state")


def test_train_jepa_from_episodic():
    """Bulk-train JEPA from an existing RainNet's episodic memory."""
    net = RainNet(config=RainNetConfig(dim=D, n_candidates=1))
    # Populate episodic via several queries.
    net.ingest_fact("fact one")
    net.ingest_fact("fact two")
    net.answer("test query one")
    net.answer("test query two")
    spec = make_jepa_expert(dim=D)
    n_steps = train_jepa_from_episodic(spec, net.memory.episodic)
    assert n_steps == 2  # one per episodic entry
    assert spec._jepa_state.n_updates == 2  # type: ignore[attr-defined]


def test_jepa_no_state_returns_zero_steps():
    """Calling train_jepa on a non-JEPA spec returns 0 (graceful)."""
    from rain.core.real_experts import make_pure_attn_expert

    spec = make_pure_attn_expert(dim=D)
    net = RainNet(config=RainNetConfig(dim=D, n_candidates=1))
    n = train_jepa_from_episodic(spec, net.memory.episodic)
    assert n == 0
