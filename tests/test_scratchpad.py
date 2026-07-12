"""Tests for the ScratchpadReasoner chain-of-thought primitive."""

from rain.agent import ConsciousAgent
from rain.cognition.scratchpad import ReasoningTrace, ScratchpadReasoner


def test_decompose_safe_with_pattern():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    r = ScratchpadReasoner(agent)
    sub = r.decompose("Is amoxicillin safe with warfarin?")
    assert sub == [("amoxicillin", "interacts_with", "warfarin")]


def test_decompose_where_does_x_live():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    r = ScratchpadReasoner(agent)
    sub = r.decompose("where does the lion live?")
    assert sub == [("lion", "lives_in", "")]


def test_decompose_what_is_capital_of():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    r = ScratchpadReasoner(agent)
    sub = r.decompose("what is the capital of france?")
    assert sub == [("france", "capital", "")]


def test_reason_with_kb_hit_returns_grounded_answer():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    agent.tell("lion", "lives_in", "savanna")
    r = ScratchpadReasoner(agent)
    trace = r.reason("where does the lion live?")
    assert isinstance(trace, ReasoningTrace)
    assert len(trace.steps) == 1
    assert "savanna" in trace.final_answer
    assert trace.final_epistemic in ("know", "think")


def test_reason_creates_scratchpad_facts_in_kb():
    """After reason() succeeds with a KB hit, scratch:* triples
    should appear in the agent's KB so KB-Attention can retrieve them."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    agent.tell("paris", "capital", "france")
    r = ScratchpadReasoner(agent)
    r.reason("what is the capital of paris?")
    # The scratchpad write should have produced a scratch:* triple
    # (don't assert exact content; just that the kb received something)
    # Use the agent's ask path with the prefixed subject.
    follow = agent.ask("scratch:paris", "scratch:capital")
    assert follow.inference_source in ("direct", "inherited", "transitive")


def test_reason_kb_miss_returns_unknown():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    r = ScratchpadReasoner(agent)
    trace = r.reason("where does the zebrafish live?")
    # KB miss + no HYMN -> step records 'unknown' epistemic
    assert len(trace.steps) == 1
    assert trace.steps[0].epistemic == "unknown"


def test_reason_returns_step_per_subquestion():
    """The 'does X eat Y' pattern produces 2 sub-questions; trace should
    have both steps logged."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    r = ScratchpadReasoner(agent)
    trace = r.reason("does the lion eat fish?")
    assert len(trace.steps) == 2
