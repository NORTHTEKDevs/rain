# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the v0.2 novel additions: reflexion loop + causal graph."""

import numpy as np
import pytest

from rain.cognition.reflexion import reflect, reflect_and_answer
from rain.core.causal_graph import CausalEdge, CausalGraph, CausalNode
from rain.core.hv_substrate import hash_to_hv, similarity
from rain.core.rain_net import RainNet, RainNetConfig


D = 1024
NET_DIM = 1024


# ============================================================
# Reflexion
# ============================================================


def _net():
    return RainNet(config=RainNetConfig(dim=NET_DIM, n_candidates=2, semantic_top_k=3))


def test_reflexion_runs_and_returns_result():
    net = _net()
    net.ingest_fact("Apollo 11 landed on the Moon in 1969.")
    result = reflect(net, "When did Apollo 11 land?", n_candidates=4, max_rounds=2)
    assert result.final_answer_hv.shape == (NET_DIM,)
    assert 1 <= len(result.rounds) <= 2
    assert all(0 <= r.chosen_idx < 4 for r in result.rounds)


def test_reflexion_records_per_round_scores():
    net = _net()
    net.ingest_fact("test fact")
    result = reflect(net, "test query", n_candidates=3, max_rounds=2)
    for r in result.rounds:
        assert len(r.candidate_scores) == 3
        # best_score should equal max of candidate_scores
        assert abs(r.best_score - max(r.candidate_scores)) < 1e-5


def test_reflexion_early_stop_on_no_improvement():
    """If improvement is below threshold, the loop should stop early."""
    net = _net()
    result = reflect(
        net,
        "any query",
        n_candidates=2,
        max_rounds=5,
        improvement_threshold=10.0,  # impossibly high; should always stop after round 1
    )
    # max_rounds=5, threshold huge -> should end at round 1 (cannot improve enough)
    assert len(result.rounds) <= 2


def test_reflect_and_answer_returns_audit_report():
    net = _net()
    net.ingest_fact("Apollo 11 landed on the Moon in 1969.", source="nasa")
    report = reflect_and_answer(net, "When did Apollo 11 land?", n_candidates=4, max_rounds=2)
    assert report.answer_text is not None
    assert any("reflexion" in w.lower() for w in report.warnings)


# ============================================================
# Causal graph
# ============================================================


def test_causal_graph_add_node_idempotent():
    g = CausalGraph(dim=D)
    n1 = g.add_node("Smoking")
    n2 = g.add_node("smoking")  # case-insensitive
    assert n1 is n2
    assert len(g.nodes) == 1


def test_causal_graph_add_edge():
    g = CausalGraph(dim=D)
    e = g.add_edge("Smoking", "causes", "Cancer")
    assert e.source == "Smoking"
    assert e.target == "Cancer"
    assert len(g.edges) == 1
    assert len(g.nodes["smoking"].outgoing) == 1
    assert len(g.nodes["cancer"].incoming) == 1


def test_causal_extract_from_text():
    g = CausalGraph(dim=D)
    n = g.add_triples_from_text(
        "Smoking causes Cancer. Stress leads to High Blood Pressure.",
        fact_id="f0",
    )
    assert n >= 2
    assert "smoking" in g.nodes
    assert "cancer" in g.nodes


def test_causal_ancestors():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "causes", "C")
    g.add_edge("X", "causes", "C")
    ancestors_of_c = g.ancestors("C", max_depth=5)
    names = {n for n, _d in ancestors_of_c}
    assert "B" in names
    assert "A" in names
    assert "X" in names


def test_causal_descendants():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "causes", "C")
    g.add_edge("A", "causes", "D")
    descendants = g.descendants("A", max_depth=5)
    names = {n for n, _d in descendants}
    assert "B" in names
    assert "C" in names
    assert "D" in names


def test_causal_chain():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "causes", "C")
    g.add_edge("C", "causes", "D")
    chain = g.causal_chain("A", "D")
    assert chain is not None
    assert len(chain) == 3
    assert chain[0].source == "A"
    assert chain[-1].target == "D"


def test_causal_chain_no_path():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("X", "causes", "Y")
    chain = g.causal_chain("A", "Y")
    assert chain is None


def test_causal_chain_missing_node():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    chain = g.causal_chain("A", "Nonexistent")
    assert chain is None


def test_causal_do_intervention():
    g = CausalGraph(dim=D)
    g.add_edge("Rain", "causes", "Wet Ground")
    g.add_edge("Wet Ground", "causes", "Slippery Roads")
    new_rain_hv = hash_to_hv("new::heavy_rain", dim=D)
    result = g.do_intervention("Rain", new_rain_hv, max_propagation_depth=3)
    assert "Rain" in result
    assert "Wet Ground" in result
    assert "Slippery Roads" in result


def test_causal_no_cycle_simple():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "causes", "C")
    assert not g.has_cycle()


def test_causal_detect_cycle():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "causes", "C")
    g.add_edge("C", "causes", "A")
    assert g.has_cycle()


def test_causal_stats():
    g = CausalGraph(dim=D)
    g.add_edge("A", "causes", "B")
    g.add_edge("B", "leads to", "C")
    s = g.stats()
    assert s["n_nodes"] == 3
    assert s["n_edges"] == 2
    assert s["has_cycle"] is False
    assert "causes" in s["relation_counts"]


def test_causal_from_kb_integration():
    """Build causal graph from a RainNet's semantic memory."""
    net = _net()
    net.ingest_fact("Smoking causes Cancer.", fact_id="f1")
    net.ingest_fact("Stress leads to High Blood Pressure.", fact_id="f2")
    net.ingest_fact("Cancer leads to Death.", fact_id="f3")
    g = CausalGraph(dim=NET_DIM)
    n = g.add_triples_from_kb(net.memory.semantic)
    assert n >= 3
    # Chain: Smoking -> Cancer -> Death
    chain = g.causal_chain("Smoking", "Death")
    assert chain is not None
    assert len(chain) == 2
