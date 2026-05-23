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


def test_seed_accepts_long_form_keys_from_ollama_pipeline(tmp_path):
    """The Ollama distillation pipeline (scripts.seed_kb_from_ollama) writes
    {subject, relation, object} + optional topic/source metadata. seed_from_jsonl
    must accept that shape directly, with no preprocessing required."""
    p = tmp_path / "facts.jsonl"
    p.write_text(
        '{"subject":"lion","relation":"lives_in","object":"savanna","topic":"lions","source":"llama3.2:3b"}\n'
        '{"subject":"lion","relation":"isa","object":"mammal","topic":"lions","source":"llama3.2:3b"}\n'
    )
    kb = ShardedKB(num_shards=4, dim=10000, seed=0)
    n = seed_from_jsonl(kb, str(p))
    assert n == 2
    assert kb.query("lion", "lives_in") == "savanna"
    assert kb.query("lion", "isa") == "mammal"


def test_seed_skips_lines_missing_a_triple(tmp_path):
    """Defense in depth: a line missing s/r/o (and missing subject/relation/object)
    is silently skipped, not an error -- the upstream validator already filtered
    these, so this is just belt-and-braces."""
    p = tmp_path / "facts.jsonl"
    p.write_text(
        '{"s":"a","r":"r","o":"b"}\n'
        '{"foo":"bar"}\n'
        '{"subject":"c","relation":"r","object":"d"}\n'
    )
    kb = ShardedKB(num_shards=4, dim=10000, seed=0)
    assert seed_from_jsonl(kb, str(p)) == 2
