from rain.cognition.explain import explain
from rain.cognition.inference import InferenceResult


def test_explain_direct():
    ir = InferenceResult(answer="italy", chain=[("rome", "capital_of", "italy")], source="direct")
    e = explain(ir)
    assert e.answer == "italy"
    assert ("rome", "capital_of", "italy") in e.citations
    assert "rome" in e.text


def test_explain_inherited():
    ir = InferenceResult(
        answer="yes",
        chain=[("sparrow", "isa", "bird"), ("bird", "can_fly", "yes")],
        source="inherited",
    )
    e = explain(ir)
    assert e.answer == "yes"
    assert "sparrow" in e.text
    assert "bird" in e.text


def test_explain_no_answer():
    ir = InferenceResult(answer=None, chain=[], source=None)
    e = explain(ir)
    assert e.answer is None
    assert e.citations == []
    assert "enough" in e.text.lower() or "don't" in e.text.lower()
