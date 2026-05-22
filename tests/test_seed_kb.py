# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from rain.core.knowledge_base import ShardedKB
from scripts.seed_kb import seed_from_jsonl

def test_seed_loads_facts(tmp_path):
    p = tmp_path / "facts.jsonl"
    p.write_text('{"s":"a","r":"r","o":"b"}\n')
    kb = ShardedKB(num_shards=4, dim=10000, seed=0)
    n = seed_from_jsonl(kb, str(p))
    assert n == 1
    assert kb.query("a", "r") == "b"
