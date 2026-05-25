# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for two-stage retrieval (fast dense shortlist -> HV rerank)."""

import pytest

from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.two_stage_retrieve import TwoStageRetriever


D = 1024


def _net() -> RainNet:
    return RainNet(config=RainNetConfig(dim=D, n_candidates=1, semantic_top_k=5))


def test_retriever_initial_state():
    r = TwoStageRetriever(net=_net())
    assert len(r._dense_bank) == 0
    assert r.retrieve("anything") == []


def test_retriever_ingest_and_retrieve():
    r = TwoStageRetriever(net=_net())
    r.ingest_fact("Apollo 11 landed on the Moon in 1969.", fact_id="apollo")
    r.ingest_fact("Python is a programming language.", fact_id="python")
    r.ingest_fact("DNA carries genetic information.", fact_id="dna")
    results = r.retrieve("when did apollo land", shortlist_k=3, final_k=2)
    assert len(results) >= 1
    top_ids = [f.fact_id for _s, f in results]
    assert "apollo" in top_ids


def test_retriever_bulk_ingest_preserves_ids():
    r = TwoStageRetriever(net=_net())
    texts = [f"fact {i} about topic {i}" for i in range(20)]
    fact_ids = [f"f{i}" for i in range(20)]
    fids_out = r.bulk_ingest(texts, fact_ids=fact_ids)
    assert fids_out == fact_ids
    assert len(r._dense_bank) == 20
    assert len(r._dense_fact_ids) == 20


def test_retriever_shortlist_smaller_than_kb():
    """When shortlist_k < KB size, we should still get results."""
    r = TwoStageRetriever(net=_net())
    for i in range(50):
        r.ingest_fact(f"fact number {i} about random thing", fact_id=f"f{i}")
    results = r.retrieve("any query", shortlist_k=10, final_k=3)
    assert 1 <= len(results) <= 3


def test_retriever_records_stats():
    r = TwoStageRetriever(net=_net())
    for i in range(10):
        r.ingest_fact(f"fact {i}", fact_id=f"f{i}")
    r.retrieve("test", shortlist_k=5, final_k=2)
    stats = r.stats()
    assert stats["n_dense_facts"] == 10
    assert stats["n_hv_facts"] == 10
    assert stats["last_query"] is not None
    assert "shortlist_ms" in stats["last_query"]


def test_retriever_answer_returns_audit_report():
    r = TwoStageRetriever(net=_net())
    r.ingest_fact("Apollo 11 landed on the Moon on July 20, 1969.")
    report = r.answer("When did Apollo 11 land?", shortlist_k=5, final_k=2)
    assert report.answer_text is not None
    assert len(report.cited_facts) >= 1
    # Should have stats warning prefix
    assert any("two_stage_retrieve" in w for w in report.warnings)


def test_retriever_empty_kb_graceful():
    r = TwoStageRetriever(net=_net())
    report = r.answer("any query")
    assert "No grounded answer" in report.answer_text


def test_retriever_quality_matches_single_stage():
    """Two-stage with reasonable shortlist should match single-stage quality."""
    r = TwoStageRetriever(net=_net())
    facts = [
        ("Apollo 11 landed on the Moon in 1969.", "apollo"),
        ("The Eiffel Tower is in Paris.", "eiffel"),
        ("Python was created by Guido van Rossum.", "python"),
        ("Mount Everest is the highest mountain.", "everest"),
        ("DNA is a double helix.", "dna"),
    ]
    for text, fid in facts:
        r.ingest_fact(text, fact_id=fid)
    queries = [
        ("when did Apollo land on the moon", "apollo"),
        ("where is the Eiffel Tower", "eiffel"),
        ("who created Python", "python"),
    ]
    hits = 0
    for q, expected in queries:
        results = r.retrieve(q, shortlist_k=5, final_k=1)
        if results and results[0][1].fact_id == expected:
            hits += 1
    assert hits >= 2, f"only {hits}/3 retrieved correctly"
