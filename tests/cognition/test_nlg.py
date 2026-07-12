from rain.cognition.nlg import render


def test_known_relation_renders_template():
    out = render("rome", "capital_of", "italy")
    assert "rome" in out and "italy" in out and "capital" in out


def test_unknown_relation_uses_fallback():
    out = render("a", "fizzles", "b")
    assert "a" in out and "b" in out and "fizzles" in out


def test_deterministic_template_selection():
    out1 = render("rome", "isa", "city")
    out2 = render("rome", "isa", "city")
    assert out1 == out2
