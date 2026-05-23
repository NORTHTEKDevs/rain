# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the read-side cognitive signals exposed on Answer."""

from rain.agent import ConsciousAgent


def test_no_cognitive_signals_when_continual_off():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    assert ans.cognitive_signals is None


def test_cognitive_signals_present_when_continual_on():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    cs = ans.cognitive_signals
    assert cs is not None
    # All three surfaces report something.
    assert "lsm_state_l2" in cs
    assert "fep_cos" in cs
    assert "tsetlin_max_abs_vote" in cs


def test_fep_cos_in_range_for_kb_hit():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    # Train a bit so FEP's A is well-defined.
    for i in range(20):
        agent.tell(f"sub_{i}", "isa", f"obj_{i}")
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    cs = ans.cognitive_signals
    # Cosine similarity should be in [-1, 1].
    assert -1.0 <= cs["fep_cos"] <= 1.0


def test_lsm_state_evolves_across_ask_calls():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    agent.tell("a", "isa", "b")
    ans1 = agent.ask("a", "isa")
    l2_after_one = ans1.cognitive_signals["lsm_state_l2"]
    # Same probe again -> LSM has stepped twice now -> state should change.
    ans2 = agent.ask("a", "isa")
    l2_after_two = ans2.cognitive_signals["lsm_state_l2"]
    assert l2_after_one != l2_after_two


def test_kb_miss_still_returns_cognitive_signals():
    """Even on KB miss, the cognitive surfaces still see the state and report
    what they can. FEP cosine is 0.0 (no target to compare); LSM state still
    evolves (the reservoir always steps on the input)."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    ans = agent.ask("unknown", "relation")
    assert ans.cognitive_signals is not None
    assert ans.cognitive_signals["fep_cos"] == 0.0  # no answer to compare
    assert ans.cognitive_signals["lsm_state_l2"] >= 0.0


def test_cognitive_agreement_in_unit_interval():
    """The blended `cognitive_agreement` score stays in [0, 1]."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    score = ans.cognitive_signals["cognitive_agreement"]
    assert 0.0 <= score <= 1.0
