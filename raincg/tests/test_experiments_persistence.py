"""Smoke tests for --out result-JSON persistence added to experiments/ scripts.

Fast checks only: --help works (CLI wiring is sound) for every script, plus one
real tiny end-to-end run (scan_hybrid_outputonly, ~6s) that validates the emitted
JSON matches the SHARED CONTRACT shape.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

SCRIPTS = [
    "experiments.scan_transformer_baseline",
    "experiments.scan_hybrid",
    "experiments.scan_hybrid_outputonly",
    "experiments.scan_llm_baseline",
    "experiments.cogs_llm_baseline",
    "experiments.scan_grammar_induction",
]

REQUIRED_KEYS = {
    "bench", "system", "split", "n", "correct", "accuracy", "ci95",
    "params", "train_s", "eval_s", "seed", "config", "timestamp",
    "evidence_tier", "notes",
}


def test_result_io_wilson_ci_and_emit(tmp_path):
    from experiments._result_io import emit_result, wilson_ci

    lo, hi = wilson_ci(8, 10)
    assert 0.0 <= lo <= hi <= 1.0

    out = tmp_path / "smoke.json"
    d = emit_result(out, bench="x", system="y", split="z", n=10, correct=8, seed=0)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == d
    assert REQUIRED_KEYS <= set(loaded)
    assert loaded["accuracy"] == 0.8


def test_all_scripts_help_works():
    for mod in SCRIPTS:
        r = subprocess.run([PY, "-m", mod, "--help"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"{mod} --help failed:\n{r.stdout}\n{r.stderr}"
        assert "--out" in r.stdout, f"{mod} --help missing --out flag"


# --- OpenRouter backend: arg parsing + sanitization (no network) -------

LLM_SCRIPTS = ["experiments.scan_llm_baseline", "experiments.cogs_llm_baseline"]


def test_llm_scripts_expose_backend_flag():
    for mod in LLM_SCRIPTS:
        r = subprocess.run([PY, "-m", mod, "--help"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        assert "--backend" in r.stdout
        assert "openrouter" in r.stdout


def test_llm_scripts_reject_unknown_backend():
    for mod in LLM_SCRIPTS:
        r = subprocess.run([PY, "-m", mod, "--backend", "not-a-backend"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
        assert r.returncode != 0
        assert "invalid choice" in r.stderr.lower()


def test_openrouter_helper_requires_api_key(monkeypatch):
    """No network call is attempted when OPENROUTER_API_KEY is unset -- the
    helper must raise before touching urllib."""
    from experiments.cogs_llm_baseline import openrouter as cogs_openrouter

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        cogs_openrouter("some/model", "prompt")


def test_openrouter_helper_retries_and_gives_up_on_repeated_failure(monkeypatch):
    """Simulate 3 consecutive 429s: helper must retry up to `retries` times,
    sleep between attempts (patched to a no-op), and return (\"\", False)
    without raising -- no real network I/O in this test."""
    import urllib.error
    from experiments import cogs_llm_baseline as mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    text, ok = mod.openrouter("some/model", "prompt", retries=3)
    assert ok is False
    assert text == ""
    assert calls["n"] == 3


def test_openrouter_helper_succeeds_on_second_attempt(monkeypatch):
    import urllib.error
    from experiments import scan_llm_baseline as mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 500, "server error", {}, None)
        return FakeResp({"choices": [{"message": {"content": "I_JUMP"}}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    text, ok = mod.openrouter("some/model", "prompt", retries=3)
    assert ok is True
    assert text == "I_JUMP"
    assert calls["n"] == 2


def test_openrouter_key_never_appears_in_request_body(monkeypatch):
    """The request body must carry only the prompt/model, never the API key --
    the key belongs in the Authorization header only."""
    from experiments import scan_llm_baseline as mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-value-12345")
    captured = {}

    class FakeResp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        captured["headers"] = req.headers
        return FakeResp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    mod.openrouter("some/model", "prompt text")

    assert b"super-secret-value-12345" not in captured["body"]
    # Authorization header (urllib title-cases header keys) carries the key instead
    assert "super-secret-value-12345" in captured["headers"].get("Authorization", "")


def test_scan_hybrid_outputonly_real_tiny_run(tmp_path):
    out = tmp_path / "scan_addprim_jump__hybrid_outputonly_reinforce__seed0.json"
    r = subprocess.run(
        [PY, "-m", "experiments.scan_hybrid_outputonly",
         "--steps", "50", "--n-eval", "20", "--seed", "0", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    d = json.loads(out.read_text())
    assert REQUIRED_KEYS <= set(d)
    assert d["bench"] == "scan_addprim_jump"
    assert d["system"] == "hybrid_outputonly_reinforce"
    assert d["seed"] == 0
    assert isinstance(d["n"], int) and d["n"] > 0
    assert 0 <= d["correct"] <= d["n"]
    assert 0.0 <= d["accuracy"] <= 1.0
    assert len(d["ci95"]) == 2
