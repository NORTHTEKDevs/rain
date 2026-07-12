"""Tests for the inspect_kb script helpers."""

import json
from pathlib import Path

from scripts.inspect_kb import _read_facts, _triple


def _write(path: Path, facts: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(f) for f in facts) + "\n", encoding="utf-8")


def test_read_facts_skips_blank_and_malformed(tmp_path):
    p = tmp_path / "facts.jsonl"
    p.write_text(
        '{"subject":"lion","relation":"isa","object":"mammal"}\n'
        "\n"
        "{ malformed line }\n"
        '{"subject":"tiger","relation":"isa","object":"mammal"}\n',
        encoding="utf-8",
    )
    facts = _read_facts(p)
    assert len(facts) == 2
    assert facts[0]["subject"] == "lion"
    assert facts[1]["subject"] == "tiger"


def test_triple_extracts_both_schemas():
    assert _triple({"subject": "a", "relation": "r", "object": "b"}) == ("a", "r", "b")
    assert _triple({"s": "a", "r": "r", "o": "b"}) == ("a", "r", "b")
    assert _triple({"foo": "bar"}) is None
    assert _triple({}) is None


def test_triple_returns_none_on_partial(tmp_path):
    """A line missing one of the keys should not produce a triple."""
    assert _triple({"subject": "a", "relation": "r"}) is None
    assert _triple({"subject": "a", "object": "b"}) is None
