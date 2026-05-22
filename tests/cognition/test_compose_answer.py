# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.core.knowledge_base import ShardedKB
from rain.cognition.compose_answer import describe


def test_describe_returns_multi_sentence():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    kb.write("rome", "capital_of", "italy")
    kb.write("rome", "locatedin", "europe")
    out = describe(kb, "rome", ["capital_of", "locatedin"])
    assert "rome" in out
    assert "italy" in out
    assert "europe" in out
    assert out.count(".") >= 2


def test_describe_unknown_entity_returns_dont_know():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    out = describe(kb, "x", ["isa"])
    assert "don't know" in out.lower() or "do not know" in out.lower()
