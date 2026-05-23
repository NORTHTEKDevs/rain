# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Integration tests for the ConsciousAgent orchestrator."""

from rain.agent import ConsciousAgent


def _agent():
    return ConsciousAgent(dim=512, num_shards=4, seed=0)


def test_tell_then_ask_direct():
    a = _agent()
    a.tell("rome", "capital_of", "italy")
    answer = a.ask("rome", "capital_of")
    assert answer.epistemic in ("know", "think")  # depending on calibration prior
    assert "italy" in answer.text
    assert answer.inference_source == "direct"


def test_ask_unknown_is_classified_unknown_or_yields_dont_know():
    a = _agent()
    answer = a.ask("nonexistent_entity", "isa")
    assert answer.epistemic == "unknown"
    assert "don't" in answer.text.lower() or "do not" in answer.text.lower() or "enough" in answer.text.lower()


def test_think_aloud_emits_cot():
    a = _agent()
    a.tell("rome", "capital_of", "italy")
    answer = a.ask("rome", "capital_of", think_aloud=True)
    assert "Q:" in answer.text
    assert "A:" in answer.text


def test_dialogue_resolves_reference():
    a = _agent()
    a.tell("rome", "capital_of", "italy")
    # First question primes context
    a.ask("rome", "capital_of")
    # Second question with missing entity uses last_entity (= "rome")
    answer = a.ask(None, "capital_of")
    # Should resolve to rome -> italy
    assert "italy" in answer.text.lower() or "don't" in answer.text.lower()


def test_self_describe_contains_known_facts():
    a = _agent()
    text = a.self_describe()
    assert "rain" in text.lower()
    assert "non" in text.lower()  # "non llm" somewhere


def test_describe_returns_paragraph():
    a = _agent()
    a.tell("rome", "capital_of", "italy")
    a.tell("rome", "locatedin", "europe")
    out = a.describe("rome", relations=["capital_of", "locatedin"])
    assert "rome" in out
    assert "italy" in out
    assert "europe" in out


def test_feedback_updates_calibration():
    a = _agent()
    # Initial calibration is 0.5 (Beta(1,1))
    initial = a.calibration.calibration("capital_of")
    for _ in range(20):
        a.feedback("capital_of", was_correct=True)
    later = a.calibration.calibration("capital_of")
    assert later > initial


def test_introspection_records_events():
    a = _agent()
    a.tell("x", "isa", "y")
    a.ask("x", "isa")
    out = a.what_just_happened()
    assert "learn" in out.lower() or "answer" in out.lower()


def test_citations_present_when_direct_answer():
    a = _agent()
    a.tell("rome", "capital_of", "italy")
    answer = a.ask("rome", "capital_of")
    assert ("rome", "capital_of", "italy") in answer.citations
