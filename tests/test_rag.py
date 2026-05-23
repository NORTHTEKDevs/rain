# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the KB-augmented sampler (no HYMN model required)."""

from rain.core.knowledge_base import ShardedKB
from rain.cognition.rag import KbAugmentedSampler


def _kb_with_facts(facts: list[tuple[str, str, str]]) -> ShardedKB:
    kb = ShardedKB(num_shards=4, dim=512, seed=0)
    for s, r, o in facts:
        kb.write(s, r, o)
    return kb


def test_candidate_subjects_extracts_alphanumeric_tokens():
    kb = ShardedKB(num_shards=2, dim=128, seed=0)
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: "")
    cands = sampler.candidate_subjects("Where does a lion live?")
    assert "lion" in cands
    assert "where" in cands
    assert "live" in cands
    # 1-2 char words filtered out
    assert "a" not in cands


def test_candidate_subjects_includes_bigrams():
    kb = ShardedKB(num_shards=2, dim=128, seed=0)
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: "")
    cands = sampler.candidate_subjects("Tell me about World War")
    assert "world" in cands
    assert "war" in cands
    assert "world_war" in cands


def test_retrieve_pulls_kb_facts_for_known_subject():
    kb = _kb_with_facts([
        ("lion", "lives_in", "savanna"),
        ("lion", "isa", "mammal"),
        ("tiger", "lives_in", "jungle"),
    ])
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: "")
    facts = sampler.retrieve("Where does the lion live?")
    triples = {(s, r, o) for s, r, o in facts}
    assert ("lion", "lives_in", "savanna") in triples
    assert ("lion", "isa", "mammal") in triples


def test_retrieve_returns_empty_for_unknown_subject():
    kb = _kb_with_facts([("lion", "lives_in", "savanna")])
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: "")
    assert sampler.retrieve("What is the meaning of dragon?") == []


def test_sample_passes_context_q_a_prompt_to_base():
    """The full prompt should be 'Context: ...\\nQ: <q>\\nA:'."""
    kb = _kb_with_facts([("lion", "lives_in", "savanna")])
    captured: list[str] = []

    def base(prompt: str, n: int) -> str:
        captured.append(prompt)
        return " savanna."

    sampler = KbAugmentedSampler(kb, base_sampler=base)
    out = sampler.sample("Where does the lion live?", n_tokens=10)
    assert len(captured) == 1
    prompt = captured[0]
    assert "Context: lion lives in savanna." in prompt
    assert "Q: Where does the lion live?" in prompt
    assert prompt.endswith("A:")
    assert out == " savanna."


def test_sample_with_no_kb_hits_still_passes_q_to_base():
    """Even with empty retrieval, the Q/A format is preserved."""
    kb = _kb_with_facts([("lion", "lives_in", "savanna")])
    captured: list[str] = []
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: captured.append(p) or "ignored")
    sampler.sample("What is the meaning of dragon?", n_tokens=10)
    prompt = captured[0]
    assert "Context:" not in prompt   # no KB facts to inject
    assert prompt.startswith("Q:")
    assert prompt.endswith("A:")


def test_max_facts_caps_retrieval():
    """Don't dump too many facts into the prompt."""
    facts = [(f"subj_{i}", "isa", f"obj_{i}") for i in range(10)]
    kb = _kb_with_facts(facts)
    sampler = KbAugmentedSampler(kb, base_sampler=lambda p, n: "", max_facts=3)
    pulled = sampler.retrieve(" ".join(f"subj_{i}" for i in range(10)))
    assert len(pulled) <= 3
