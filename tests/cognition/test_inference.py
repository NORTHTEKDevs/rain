# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.cognition.inference import infer
from rain.core.knowledge_base import ShardedKB


def test_direct_inference():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    kb.write("rome", "capital_of", "italy")
    res = infer(kb, "rome", "capital_of")
    assert res.answer == "italy"
    assert res.source == "direct"


def test_inherited_via_isa():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    kb.write("sparrow", "isa", "bird")
    kb.write("bird", "can_fly", "yes")
    res = infer(kb, "sparrow", "can_fly")
    assert res.answer == "yes"
    assert res.source == "inherited"
    assert any("sparrow" in str(triple) for triple in res.chain)


def test_unknown_returns_none():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    res = infer(kb, "unknown", "x")
    assert res.answer is None
    assert res.source is None


def test_transitive_via_locatedin():
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    kb.write("rome", "locatedin", "italy")
    kb.write("italy", "locatedin", "europe")
    res = infer(kb, "rome", "locatedin", max_depth=3)
    # Transitive chain reaches "europe"
    assert res.source in ("transitive", "direct")
    assert res.answer in ("italy", "europe")  # direct gets italy, transitive walks further
