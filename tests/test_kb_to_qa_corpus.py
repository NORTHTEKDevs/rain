"""Tests for the KB -> Q/A corpus renderer."""

import random

from scripts.kb_to_qa_corpus import _capitalize_first, _deunderscore, _render_fact


def test_deunderscore_handles_basic_cases():
    assert _deunderscore("hello_world") == "hello world"
    assert _deunderscore("  trimmed_  ") == "trimmed"
    assert _deunderscore("") == ""


def test_capitalize_first_does_not_break_empty():
    assert _capitalize_first("") == ""
    assert _capitalize_first("a") == "A"
    assert _capitalize_first("hello") == "Hello"
    assert _capitalize_first("HELLO") == "HELLO"


def test_render_fact_produces_n_phrasings():
    rng = random.Random(0)
    fact = {"subject": "lion", "relation": "lives_in", "object": "savanna"}
    out = _render_fact(fact, n_phrasings=3, rng=rng)
    assert len(out) == 3
    for line in out:
        assert "lion" in line.lower()
        assert "savanna" in line
        assert line.startswith("Q:")
        assert "\nA:" in line


def test_render_fact_handles_long_form_keys():
    """Both {subject,relation,object} and {s,r,o} should render identically
    when fed the same RNG state."""
    fact_long = {"subject": "a", "relation": "r", "object": "b"}
    fact_short = {"s": "a", "r": "r", "o": "b"}
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    out_a = _render_fact(fact_long, 2, rng_a)
    out_b = _render_fact(fact_short, 2, rng_b)
    assert out_a == out_b


def test_render_fact_skips_incomplete():
    rng = random.Random(0)
    assert _render_fact({"subject": "x"}, 2, rng) == []
    assert _render_fact({}, 2, rng) == []


def test_render_fact_underscores_de_undersored_in_output():
    """The whole point of this corpus is judge-friendly natural language --
    no snake_case should appear in the rendered Q/A pair."""
    rng = random.Random(0)
    fact = {"subject": "albert_einstein", "relation": "born_in", "object": "ulm_germany"}
    out = _render_fact(fact, n_phrasings=5, rng=rng)
    for line in out:
        assert "_" not in line
        assert "albert einstein" in line.lower()
        assert "born in" in line.lower()
        assert "ulm germany" in line.lower()
