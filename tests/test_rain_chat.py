"""Tests for the rain_chat REPL parsing/dispatch logic.

The live REPL needs a ConsciousAgent and optional HYMN sampler + Ollama
judge. We exercise the pure-Python pieces (_handle_ask + the
non-interactive smoke path) without hitting the daemon.
"""

from rain.agent import ConsciousAgent
from scripts.rain_chat import _handle_ask


def test_handle_ask_returns_false_for_one_token():
    agent = ConsciousAgent(dim=256, num_shards=4, seed=0)
    assert _handle_ask(agent, "lion") is False
    assert _handle_ask(agent, "") is False


def test_handle_ask_returns_false_for_unknown_fact():
    agent = ConsciousAgent(dim=256, num_shards=4, seed=0)
    # Agent has only the SelfModel pre-seeded; asking about unknown returns no kb hit.
    assert _handle_ask(agent, "lion lives_in") is False


def test_handle_ask_returns_true_after_tell(capsys):
    """After agent.tell, the same (s, r) probe should return True."""
    agent = ConsciousAgent(dim=512, num_shards=8, seed=0)
    agent.tell("lion", "lives_in", "savanna")
    assert _handle_ask(agent, "lion lives_in") is True
    captured = capsys.readouterr()
    # The handler prints the canonical [KB] ... + citations + epistemic block.
    assert "[KB]" in captured.out
    assert "savanna" in captured.out
    assert "epistemic" in captured.out


def test_handle_ask_joins_multi_token_relation(capsys):
    """`<subject> <multi token rel>` -> relation underscored."""
    agent = ConsciousAgent(dim=512, num_shards=8, seed=0)
    agent.tell("water", "boils_at", "100c")
    assert _handle_ask(agent, "water boils at") is True
    captured = capsys.readouterr()
    assert "100c" in captured.out
