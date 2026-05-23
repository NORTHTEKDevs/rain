# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the HYMN sampler fallback wired into ConsciousAgent.ask."""

from rain.agent import ConsciousAgent


class _FakeSampler:
    """Records every call so tests can assert what the agent passed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.return_value: str = "made_up_answer"

    def __call__(self, prompt: str, n_tokens: int) -> str:
        self.calls.append((prompt, n_tokens))
        return self.return_value


def _agent() -> ConsciousAgent:
    return ConsciousAgent(dim=256, num_shards=4, seed=0)


def test_no_sampler_attached_returns_idk_on_kb_miss():
    agent = _agent()
    ans = agent.ask("lion", "lives_in")
    assert "don't know" in ans.text.lower()
    assert ans.epistemic == "unknown"
    assert ans.inference_source is None


def test_sampler_fires_on_kb_miss():
    agent = _agent()
    sampler = _FakeSampler()
    agent.attach_hymn_sampler(sampler, n_tokens=42)
    ans = agent.ask("lion", "lives_in")
    assert sampler.calls == [("lion lives in ", 42)]
    assert ans.text == "lion lives in made_up_answer"
    assert ans.epistemic == "guess"
    assert ans.inference_source == "hymn"
    assert ans.confidence == 0.3


def test_sampler_does_not_fire_on_kb_hit():
    agent = _agent()
    sampler = _FakeSampler()
    agent.attach_hymn_sampler(sampler)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    # KB hit takes precedence
    assert sampler.calls == []
    assert ans.epistemic == "think"
    assert ans.inference_source == "direct"
    assert "savanna" in ans.text


def test_sampler_does_not_mutate_calibration():
    """HYMN guess shouldn't auto-mark the relation as correct/incorrect.
    Calibration is for KB-grounded answers + explicit feedback only.
    """
    agent = _agent()
    sampler = _FakeSampler()
    agent.attach_hymn_sampler(sampler)
    pre = agent.calibration.calibration("lives_in")
    _ = agent.ask("lion", "lives_in")
    post = agent.calibration.calibration("lives_in")
    assert pre == post


def test_sampler_replaceable():
    """attach_hymn_sampler can be called again to swap samplers."""
    agent = _agent()
    s1 = _FakeSampler()
    s1.return_value = "first"
    s2 = _FakeSampler()
    s2.return_value = "second"
    agent.attach_hymn_sampler(s1)
    a1 = agent.ask("foo", "bar")
    agent.attach_hymn_sampler(s2)
    a2 = agent.ask("foo2", "bar2")
    assert "first" in a1.text
    assert "second" in a2.text
    assert len(s1.calls) == 1
    assert len(s2.calls) == 1
