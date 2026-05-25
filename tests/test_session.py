# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the multi-turn Session abstraction."""

import pytest

from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.session import Session, Turn


D = 2048


def _net() -> RainNet:
    return RainNet(config=RainNetConfig(dim=D, n_candidates=2, semantic_top_k=3))


def test_session_initial_state():
    s = Session(net=_net())
    assert len(s.turns) == 0
    assert s.recall("anything") == []


def test_session_turn_records():
    s = Session(net=_net())
    r = s.turn("hello world")
    assert len(s.turns) == 1
    assert s.turns[0].turn_id == 0
    assert s.turns[0].query == "hello world"


def test_session_ids_increment():
    s = Session(net=_net())
    s.turn("a")
    s.turn("b")
    s.turn("c")
    assert [t.turn_id for t in s.turns] == [0, 1, 2]


def test_session_recall_returns_similar_past_turn():
    s = Session(net=_net())
    s.turn("Tell me about photosynthesis in plants.")
    for _ in range(5):
        s.turn("unrelated padding question about quantum physics formulas")
    recall = s.recall("photosynthesis", top_k=1)
    assert len(recall) == 1
    # The photosynthesis turn should be top-1.
    assert "photosynthesis" in recall[0][1].query.lower()


def test_session_recall_top_k_bounded():
    s = Session(net=_net())
    for i in range(5):
        s.turn(f"question number {i} about thing {i}")
    recall = s.recall("question", top_k=3)
    assert len(recall) <= 3


def test_session_recall_at_capacity_doesnt_crash():
    s = Session(net=_net())
    # Fewer turns than top_k requested.
    s.turn("only turn")
    recall = s.recall("anything", top_k=10)
    assert len(recall) == 1


def test_session_history_summary():
    s = Session(net=_net())
    s.turn("first")
    s.turn("second")
    summary = s.history_summary()
    assert "2 turn" in summary
    assert "first" in summary
    assert "second" in summary


def test_session_history_empty():
    s = Session(net=_net())
    assert "empty" in s.history_summary().lower()


def test_session_records_recall_in_warnings():
    s = Session(net=_net())
    s.turn("Tell me about photosynthesis in plants.")
    r = s.turn("Tell me more about photosynthesis.")
    # The second turn should have recalled the first.
    recall_warning_present = any("recalled" in w.lower() for w in r.warnings)
    assert recall_warning_present


def test_session_stats():
    s = Session(net=_net())
    s.turn("a")
    s.turn("b")
    stats = s.stats()
    assert stats["n_turns"] == 2
    assert stats["episodic_records"] == 2  # one per turn via net.answer


def test_session_continual_learning_with_ingest():
    """Add a fact mid-session; later turn must retrieve it."""
    s = Session(net=_net())
    s.turn("Hello")
    s.net.ingest_fact("Apollo 11 landed on the Moon on July 20, 1969.")
    r = s.turn("When did Apollo 11 land?")
    # Should cite the newly-added fact.
    assert any("Apollo" in f.text for f in r.cited_facts)


def test_session_long_horizon_recall():
    """Even after 20 turns, an early turn should be recallable by HV similarity."""
    s = Session(net=_net())
    s.turn("Tell me about the great pyramid of Giza.")
    for i in range(20):
        s.turn(f"Tell me about random topic {i} like volcanoes or oceans.")
    recall = s.recall("pyramid", top_k=3)
    # The pyramid query (turn 0) should be near top.
    top_queries = [t.query for _s, t in recall]
    assert any("pyramid" in q.lower() for q in top_queries)
