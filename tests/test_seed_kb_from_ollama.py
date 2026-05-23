# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Schema + parsing tests for the Ollama-driven KB seed script.

These tests do NOT call the Ollama daemon -- they exercise the pure-Python
parsing + validation helpers. An integration test against a real daemon
lives outside the default suite.
"""

from scripts.seed_kb_from_ollama import (
    _extract_json_array,
    _valid_triple,
    _normalize_token,
    build_user_prompt,
)


def test_normalize_token_basic_cases():
    assert _normalize_token("Lion") == "lion"
    assert _normalize_token("  Roman Empire ") == "roman_empire"
    assert _normalize_token("World-War II") == "world_war_ii"
    # Garbage leading char + internal whitespace gets cleaned; leading "_"
    # is stripped so the result is a valid downstream KB token.
    assert _normalize_token("® nasty\nstuff!") == "nasty_stuff"


def test_valid_triple_accepts_clean_lists():
    assert _valid_triple(["lion", "isa", "mammal"]) == ("lion", "isa", "mammal")
    assert _valid_triple(["Roman Empire", "founded_in", "27 BC"]) == (
        "roman_empire", "founded_in", "27_bc",
    )


def test_valid_triple_rejects_malformed():
    assert _valid_triple(["lion", "isa"]) is None  # wrong arity
    assert _valid_triple(["lion", "", "mammal"]) is None  # empty token
    assert _valid_triple("not even a list") is None
    assert _valid_triple([1, 2, 3]) is None  # non-string


def test_extract_json_array_finds_bare_array():
    txt = '[["lion","isa","mammal"],["tiger","isa","mammal"]]'
    arr = _extract_json_array(txt)
    assert isinstance(arr, list) and len(arr) == 2


def test_extract_json_array_finds_inside_prose():
    txt = (
        "Sure! Here are the triples:\n"
        '[["lion","isa","mammal"],["tiger","isa","mammal"]]\n'
        "Hope this helps."
    )
    arr = _extract_json_array(txt)
    assert arr == [["lion", "isa", "mammal"], ["tiger", "isa", "mammal"]]


def test_extract_json_array_finds_inside_markdown_fence():
    txt = (
        "```json\n"
        '[["lion","isa","mammal"]]\n'
        "```"
    )
    arr = _extract_json_array(txt)
    assert arr == [["lion", "isa", "mammal"]]


def test_extract_json_array_returns_none_on_garbage():
    assert _extract_json_array("just prose, no json") is None
    assert _extract_json_array("[ not parseable ]") is None


def test_extract_json_array_recovers_from_missing_closing_bracket():
    """llama3.2:3b sometimes drops the outer ']'. The extractor must still
    parse the inner triples by balancing brackets."""
    txt = (
        '[\n'
        '  ["lion", "kind", "felidae"],\n'
        '  ["lion", "color", "golden_brown"],\n'
        '  ["lion", "lives_in", "savanna"]\n'
    )
    arr = _extract_json_array(txt)
    assert isinstance(arr, list) and len(arr) == 3
    assert arr[0] == ["lion", "kind", "felidae"]


def test_extract_json_array_recovers_from_trailing_partial_triple():
    """Recovery should also drop a half-written final triple."""
    txt = (
        '[\n'
        '  ["lion", "kind", "felidae"],\n'
        '  ["lion", "color", "golden_brown"],\n'
        '  ["lion", "lives_in",'  # cut off mid-triple, no value, no closing
    )
    arr = _extract_json_array(txt)
    assert isinstance(arr, list) and len(arr) == 2


def test_user_prompt_includes_topic_and_count():
    p = build_user_prompt("ancient Egypt", 25)
    assert "ancient Egypt" in p
    assert "25" in p
    assert "JSON" in p
