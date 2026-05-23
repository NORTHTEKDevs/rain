# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.cognition.inference import InferenceResult
from rain.cognition.think_aloud import narrate


def test_narrate_direct():
    ir = InferenceResult(answer="italy", chain=[("rome", "capital_of", "italy")], source="direct")
    out = narrate("what is the capital of italy?", ir)
    assert "Q:" in out
    assert "A: italy" in out
    assert "rome" in out


def test_narrate_unknown():
    ir = InferenceResult(answer=None, chain=[], source=None)
    out = narrate("who is the king of mars?", ir)
    assert "A: I don't know." in out
