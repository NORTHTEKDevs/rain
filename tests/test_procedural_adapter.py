"""Tests for procedural_adapter: skill load/save/invoke/registry."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rain.core.encoder_bank import EncoderBank
from rain.core.procedural_adapter import (
    AdapterMeta,
    ProceduralAdapter,
    SkillRegistry,
)


D = 1024


def test_from_text_builds_adapter():
    a = ProceduralAdapter.from_text(
        name="math",
        description="solves arithmetic",
        domain_text="what is, compute, evaluate, calculate, plus, minus",
        invoke_fn=lambda q, kb: "42",
        dim=D,
    )
    assert a.meta.name == "math"
    assert a.domain_hv.shape == (D,)


def test_match_returns_true_for_similar():
    enc = EncoderBank(dim=D)
    a = ProceduralAdapter.from_text(
        name="math",
        description="arithmetic",
        domain_text="compute calculate evaluate arithmetic",
        invoke_fn=lambda q, kb: "ok",
        dim=D,
    )
    # Query that shares words with domain
    q_hv = enc.encode("text", "compute the arithmetic result")
    assert a.matches(q_hv, threshold=0.1)


def test_match_returns_false_for_unrelated():
    enc = EncoderBank(dim=D)
    a = ProceduralAdapter.from_text(
        name="math",
        description="arithmetic",
        domain_text="compute calculate arithmetic",
        invoke_fn=lambda q, kb: "ok",
        dim=D,
    )
    q_hv = enc.encode("text", "what color is the apollo astronaut helmet")
    assert not a.matches(q_hv, threshold=0.3)


def test_invoke_calls_handler():
    a = ProceduralAdapter.from_text(
        name="echo",
        description="echo skill",
        domain_text="echo",
        invoke_fn=lambda q, kb: f"echo: {q}",
        dim=D,
    )
    assert a.invoke("hello") == "echo: hello"
    assert a.meta.invocations == 1


def test_invoke_no_handler_returns_stub():
    a = ProceduralAdapter.from_text(
        name="empty",
        description="no handler",
        domain_text="test",
        invoke_fn=None,
        dim=D,
    )
    out = a.invoke("anything")
    assert "no handler" in out


def test_invoke_handler_exception_returns_error_string():
    def bad(q, kb):
        raise RuntimeError("boom")
    a = ProceduralAdapter.from_text(
        name="bad",
        description="raises",
        domain_text="test",
        invoke_fn=bad,
        dim=D,
    )
    out = a.invoke("anything")
    assert "error" in out.lower()
    assert "boom" in out


def test_save_and_load_roundtrip():
    a = ProceduralAdapter.from_text(
        name="roundtrip",
        description="round-trip test",
        domain_text="testing",
        invoke_fn=None,
        dim=D,
    )
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "roundtrip"
        a.save(skill_dir)
        assert (skill_dir / "meta.json").exists()
        assert (skill_dir / "domain_hv.npy").exists()
        # Reload
        b = ProceduralAdapter.load(skill_dir)
        assert b.meta.name == "roundtrip"
        assert b.meta.description == "round-trip test"
        assert np.array_equal(a.domain_hv, b.domain_hv)


def test_save_load_with_handler_file():
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "skill_with_handler"
        skill_dir.mkdir()
        # Write a handler.py
        (skill_dir / "handler.py").write_text(
            "def invoke(query_text, kb):\n    return f'handled: {query_text}'\n",
            encoding="utf-8",
        )
        # Build + save the rest
        a = ProceduralAdapter.from_text(
            name="handled",
            description="has handler",
            domain_text="test",
            invoke_fn=None,
            dim=D,
        )
        a.save(skill_dir)
        # Reload -- the handler.py should now be picked up
        b = ProceduralAdapter.load(skill_dir)
        assert b.invoke_fn is not None
        assert b.invoke("hello", kb=None) == "handled: hello"


def test_save_load_with_weights():
    a = ProceduralAdapter.from_text(
        name="with_weights",
        description="has weights",
        domain_text="test",
        invoke_fn=None,
        dim=D,
    )
    a._weights["w0"] = np.random.randn(8, 8).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "with_weights"
        a.save(skill_dir)
        assert (skill_dir / "weights.npz").exists()
        b = ProceduralAdapter.load(skill_dir)
        assert "w0" in b._weights
        assert np.array_equal(a._weights["w0"], b._weights["w0"])


def test_registry_load_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(3):
            a = ProceduralAdapter.from_text(
                name=f"skill_{i}",
                description=f"skill {i}",
                domain_text=f"domain text for skill {i}",
                invoke_fn=lambda q, kb, i=i: f"skill {i} replies",
                dim=D,
            )
            a.save(root / f"skill_{i}")
        reg = SkillRegistry()
        n = reg.load_dir(root)
        assert n == 3
        assert len(reg) == 3


def test_registry_match():
    enc = EncoderBank(dim=D)
    reg = SkillRegistry()
    reg.register(
        ProceduralAdapter.from_text(
            name="math", description="arithmetic",
            domain_text="compute calculate arithmetic",
            invoke_fn=lambda q, kb: "math",
            dim=D,
        )
    )
    reg.register(
        ProceduralAdapter.from_text(
            name="weather", description="weather lookup",
            domain_text="forecast weather temperature",
            invoke_fn=lambda q, kb: "sunny",
            dim=D,
        )
    )
    q_hv = enc.encode("text", "compute the arithmetic")
    matches = reg.match(q_hv, threshold=0.1)
    assert len(matches) >= 1
    assert matches[0][1].meta.name == "math"


def test_registry_empty_dir_returns_zero():
    with tempfile.TemporaryDirectory() as tmp:
        reg = SkillRegistry()
        assert reg.load_dir(tmp) == 0


def test_registry_missing_dir_returns_zero():
    reg = SkillRegistry()
    assert reg.load_dir("/nonexistent/path/that/does/not/exist") == 0
