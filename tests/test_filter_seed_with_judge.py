# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Unit tests for the seed filter (live judge patched out)."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.filter_seed_with_judge import (
    _format_answer,
    _format_question,
    _triple_from,
    run,
)


@dataclass
class _StubVerdict:
    correct: bool
    confidence: float
    reasoning: str = ""
    raw: str = ""


class _StubJudge:
    """Returns a programmed sequence of verdicts in order."""

    def __init__(self, verdicts: list[_StubVerdict | None]):
        self.verdicts = list(verdicts)
        self.calls = 0

    def judge(self, q, a):
        v = self.verdicts[self.calls % len(self.verdicts)]
        self.calls += 1
        return v


def _write_jsonl(path: Path, facts: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(f) for f in facts) + "\n")


def test_triple_from_handles_both_shapes():
    assert _triple_from({"subject": "a", "relation": "r", "object": "b"}) == ("a", "r", "b")
    assert _triple_from({"s": "a", "r": "r", "o": "b"}) == ("a", "r", "b")
    assert _triple_from({}) is None
    assert _triple_from({"subject": "a"}) is None


def test_format_question_and_answer_are_human_readable():
    assert _format_question("lion", "lives_in") == "What is the lives_in of lion?"
    assert _format_answer("lion", "lives_in", "savanna") == "The lives_in of lion is savanna."


def test_run_keeps_only_correct_verdicts(tmp_path, monkeypatch):
    """4 facts in; judge says T/F/T/F (all conf=0.9). Expect 2 kept, 2 rejected."""
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    rej_path = tmp_path / "rej.jsonl"
    _write_jsonl(
        in_path,
        [
            {"subject": "lion", "relation": "isa", "object": "mammal"},
            {"subject": "soccer", "relation": "made_of", "object": "feet"},
            {"subject": "owl", "relation": "isa", "object": "bird"},
            {"subject": "cell", "relation": "lives_in", "object": "organism"},
        ],
    )
    import scripts.filter_seed_with_judge as mod

    monkeypatch.setattr(
        mod,
        "OllamaJudge",
        lambda **k: _StubJudge(
            [
                _StubVerdict(True, 0.9),
                _StubVerdict(False, 0.9),
                _StubVerdict(True, 0.9),
                _StubVerdict(False, 0.9),
            ]
        ),
    )
    args = argparse.Namespace(
        in_path=str(in_path),
        out_path=str(out_path),
        rejected_path=str(rej_path),
        judge_model="stub",
        min_confidence=0.5,
        accept_unsure=False,
        url="http://stub",
        timeout=5,
        verbose=False,
    )
    rs = run(args)
    assert rs.total == 4
    assert rs.judged == 4
    assert rs.kept == 2
    assert rs.rejected == 2
    kept_lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert {row["subject"] for row in kept_lines} == {"lion", "owl"}
    rej_lines = [json.loads(line) for line in rej_path.read_text().splitlines()]
    assert {row["subject"] for row in rej_lines} == {"soccer", "cell"}


def test_unsure_verdicts_drop_by_default(tmp_path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"subject": "a", "relation": "r", "object": "b"}])
    import scripts.filter_seed_with_judge as mod

    monkeypatch.setattr(mod, "OllamaJudge", lambda **k: _StubJudge([_StubVerdict(True, 0.1)]))
    args = argparse.Namespace(
        in_path=str(in_path),
        out_path=str(out_path),
        rejected_path=None,
        judge_model="stub",
        min_confidence=0.5,
        accept_unsure=False,
        url="http://stub",
        timeout=5,
        verbose=False,
    )
    rs = run(args)
    assert rs.kept == 0 and rs.rejected == 1


def test_unsure_verdicts_kept_when_accept_unsure(tmp_path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"subject": "a", "relation": "r", "object": "b"}])
    import scripts.filter_seed_with_judge as mod

    monkeypatch.setattr(mod, "OllamaJudge", lambda **k: _StubJudge([_StubVerdict(True, 0.1)]))
    args = argparse.Namespace(
        in_path=str(in_path),
        out_path=str(out_path),
        rejected_path=None,
        judge_model="stub",
        min_confidence=0.5,
        accept_unsure=True,
        url="http://stub",
        timeout=5,
        verbose=False,
    )
    rs = run(args)
    assert rs.kept == 1 and rs.rejected == 0


def test_judge_parse_failure_is_kept_with_annotation(tmp_path, monkeypatch):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"subject": "a", "relation": "r", "object": "b"}])
    import scripts.filter_seed_with_judge as mod

    monkeypatch.setattr(mod, "OllamaJudge", lambda **k: _StubJudge([None]))
    args = argparse.Namespace(
        in_path=str(in_path),
        out_path=str(out_path),
        rejected_path=None,
        judge_model="stub",
        min_confidence=0.5,
        accept_unsure=False,
        url="http://stub",
        timeout=5,
        verbose=False,
    )
    rs = run(args)
    assert rs.judge_parse_failures == 1
    assert rs.kept == 1
    kept_row = json.loads(out_path.read_text().strip())
    assert kept_row["judge"]["status"] == "parse_failure"
