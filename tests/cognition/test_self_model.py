# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.cognition.self_model import SelfModel, DEFAULT_SELF_FACTS


def test_default_facts_loaded():
    sm = SelfModel(dim=512, num_shards=4, seed=0)
    assert sm.query("rain", "category") == "non_llm"
    assert sm.query("rain", "creator") == "kristian_baer"


def test_describe_returns_all_facts_as_sentences():
    sm = SelfModel(dim=512, num_shards=4, seed=0)
    sentences = sm.describe()
    assert len(sentences) == len(DEFAULT_SELF_FACTS)
    assert any("non llm" in s for s in sentences)


def test_add_fact_then_query():
    sm = SelfModel(dim=512, num_shards=4, seed=0)
    sm.add_fact("rain", "version", "0_0_3")
    assert sm.query("rain", "version") == "0_0_3"
