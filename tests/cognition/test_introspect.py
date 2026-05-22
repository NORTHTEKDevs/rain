# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.cognition.introspect import Introspector, Broadcast


def test_record_and_recent():
    intr = Introspector(capacity=4)
    intr.record("ask", question="who am I?")
    intr.record("answer", answer="non-LLM generative AI")
    assert len(intr.recent()) == 2


def test_capacity_eviction():
    intr = Introspector(capacity=2)
    intr.record("ask", question="q1")
    intr.record("ask", question="q2")
    intr.record("ask", question="q3")
    recent = intr.recent()
    assert len(recent) == 2
    assert recent[0].payload["question"] == "q2"


def test_narrate_includes_all_kinds():
    intr = Introspector(capacity=10)
    intr.record("ask", question="q")
    intr.record("answer", answer="a")
    intr.record("learn", fact="f")
    intr.record("refuse", reason="r")
    intr.record("tool", name="t")
    intr.record("explain", topic="o")
    out = intr.narrate()
    for kw in ["asked", "answered", "learned", "refused", "tool", "explained"]:
        assert kw in out
