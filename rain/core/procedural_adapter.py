# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Procedural-memory adapter format: on-disk compiled skills for RAIN-Net.

In the RAIN-Net architecture, "skills" are compiled procedures stored
as small adapter weights, keyed by an HV pattern. When the router's
similarity score against a skill's domain HV is high, the skill is
loaded and invoked.

This module defines the on-disk format + load/save/invoke utilities.
A skill is a self-contained directory:

    skill_name/
      meta.json          name, description, domain_hv path, version
      domain_hv.npy      (D,) float32 -- the routing key
      weights.npz        adapter weights (optional, can be empty)
      handler.py         optional callable: invoke(query_text, kb) -> str

The handler.py is the simplest form -- a pure Python function that
takes query text + a KB reference and returns answer text. More complex
skills can load weights into a torch model in handler.

Load:
    skill = ProceduralAdapter.load("skills/math_solver")
    skill.matches(query_hv)        # routing decision
    answer = skill.invoke("compute 5+3", kb=net.memory.semantic)

Save (for skill authoring):
    skill = ProceduralAdapter(
        name="math_solver",
        description="solves arithmetic",
        domain_text="what is, compute, evaluate",
        invoke_fn=lambda q, kb: str(eval(q.split()[-1])),
    )
    skill.save("skills/math_solver")
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hv_substrate import DEFAULT_DIM, similarity


@dataclass
class AdapterMeta:
    """Metadata persisted to skill_dir/meta.json."""

    name: str
    description: str
    version: str = "0.1.0"
    domain_text: str = ""  # the text used to encode domain_hv (for human ref)
    dim: int = DEFAULT_DIM
    has_handler: bool = False
    has_weights: bool = False
    author: str = ""
    created: str = ""  # ISO timestamp
    invocations: int = 0
    # Pattern-based fast-path triggers. If the query text matches any
    # of these regex patterns, the skill's effective match score gets
    # boosted by +0.5 (clamped to <=1.0). Useful for skills that have
    # an unmistakable surface signal -- e.g. math queries contain digits.
    trigger_patterns: list[str] = field(default_factory=list)


@dataclass
class ProceduralAdapter:
    """One compiled skill, loaded from disk OR built in-memory.

    Skill matching: cosine similarity between query_hv and the skill's
    domain_hv, gated by a threshold. Optionally boosted by a regex
    trigger pattern match on the raw query text (for skills where the
    surface form is unmistakable, e.g. arithmetic).

    Skill invocation: calls the loaded handler with the query text and
    an optional KB reference. Returns the handler's answer string.
    """

    meta: AdapterMeta
    domain_hv: np.ndarray
    invoke_fn: Callable[[str, Any], str] | None = None
    _weights: dict[str, np.ndarray] = field(default_factory=dict)

    def matches(self, query_hv: np.ndarray, threshold: float = 0.3) -> bool:
        """Routing decision: does this skill apply to the query?"""
        return similarity(self.domain_hv, query_hv) >= threshold

    def match_score(self, query_hv: np.ndarray, query_text: str = "") -> float:
        """Cosine score, plus an optional pattern-trigger boost (+0.5)
        if any trigger_pattern matches query_text. Clamped to [0, 1]."""
        base = similarity(self.domain_hv, query_hv)
        if query_text and self.meta.trigger_patterns:
            import re

            for pat in self.meta.trigger_patterns:
                try:
                    if re.search(pat, query_text):
                        return min(1.0, base + 0.5)
                except re.error:
                    continue
        return base

    def invoke(self, query_text: str, kb: Any = None) -> str:
        """Run the skill on a query. Returns answer text.

        If no handler is wired, returns a default "skill present but
        no handler" message -- useful for skill stubs in testing.
        """
        self.meta.invocations += 1
        if self.invoke_fn is None:
            return f"[skill {self.meta.name} has no handler]"
        try:
            return self.invoke_fn(query_text, kb)
        except Exception as e:  # noqa: BLE001
            return f"[skill {self.meta.name} error: {e}]"

    # -------- save / load --------

    def save(self, skill_dir: str | Path) -> None:
        """Persist the skill to a directory."""
        d = Path(skill_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.meta.has_handler = self.invoke_fn is not None
        self.meta.has_weights = bool(self._weights)
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "name": self.meta.name,
                    "description": self.meta.description,
                    "version": self.meta.version,
                    "domain_text": self.meta.domain_text,
                    "dim": self.meta.dim,
                    "has_handler": self.meta.has_handler,
                    "has_weights": self.meta.has_weights,
                    "author": self.meta.author,
                    "created": self.meta.created,
                    "invocations": self.meta.invocations,
                    "trigger_patterns": self.meta.trigger_patterns,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(d / "domain_hv.npy", self.domain_hv.astype(np.float32))
        if self._weights:
            np.savez(d / "weights.npz", **self._weights)

    @classmethod
    def load(cls, skill_dir: str | Path) -> "ProceduralAdapter":
        """Load a skill from disk. Imports handler.py if present."""
        d = Path(skill_dir)
        meta_dict = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta = AdapterMeta(
            name=meta_dict["name"],
            description=meta_dict["description"],
            version=meta_dict.get("version", "0.1.0"),
            domain_text=meta_dict.get("domain_text", ""),
            dim=meta_dict.get("dim", DEFAULT_DIM),
            has_handler=meta_dict.get("has_handler", False),
            has_weights=meta_dict.get("has_weights", False),
            author=meta_dict.get("author", ""),
            created=meta_dict.get("created", ""),
            invocations=meta_dict.get("invocations", 0),
            trigger_patterns=meta_dict.get("trigger_patterns", []),
        )
        domain_hv = np.load(d / "domain_hv.npy").astype(np.float32)
        weights: dict[str, np.ndarray] = {}
        w_path = d / "weights.npz"
        if w_path.exists():
            with np.load(w_path) as data:
                weights = {k: data[k] for k in data.files}
        invoke_fn: Callable[[str, Any], str] | None = None
        h_path = d / "handler.py"
        if h_path.exists():
            invoke_fn = _load_handler(h_path)
        adapter = cls(
            meta=meta,
            domain_hv=domain_hv,
            invoke_fn=invoke_fn,
            _weights=weights,
        )
        return adapter

    # -------- factory builders --------

    @classmethod
    def from_text(
        cls,
        name: str,
        description: str,
        domain_text: str,
        invoke_fn: Callable[[str, Any], str] | None = None,
        dim: int = DEFAULT_DIM,
        author: str = "",
        trigger_patterns: list[str] | None = None,
    ) -> "ProceduralAdapter":
        """Build an adapter from a human-readable domain text + handler fn.

        The domain_text is encoded via the standard encoder bank, so the
        resulting domain_hv lives in the shared HV space.
        """
        import time as _time

        enc = EncoderBank(dim=dim)
        domain_hv = enc.encode("text", domain_text)
        meta = AdapterMeta(
            name=name,
            description=description,
            version="0.1.0",
            domain_text=domain_text,
            dim=dim,
            has_handler=invoke_fn is not None,
            has_weights=False,
            author=author,
            created=_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trigger_patterns=list(trigger_patterns or []),
        )
        return cls(meta=meta, domain_hv=domain_hv, invoke_fn=invoke_fn)


# -------- internal handler loader --------


def _load_handler(handler_path: Path) -> Callable[[str, Any], str] | None:
    """Dynamically import handler.py and pull out its `invoke` function.

    handler.py must define a top-level function:
        def invoke(query_text: str, kb) -> str: ...
    Anything else (imports etc.) is OK.
    """
    spec = importlib.util.spec_from_file_location(
        f"_rain_skill_{handler_path.parent.name}", handler_path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Make sure the module can find its own helpers (no path pollution
    # of sys.modules).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(module, "invoke", None)
    if not callable(fn):
        return None
    return fn


# -------- registry --------


@dataclass
class SkillRegistry:
    """An in-process registry of loaded adapters. RainNet checks this
    at routing time as a complement to the procedural-memory bank in
    hierarchical_memory.
    """

    skills: list[ProceduralAdapter] = field(default_factory=list)

    def register(self, adapter: ProceduralAdapter) -> None:
        self.skills.append(adapter)

    def load_dir(self, root_dir: str | Path) -> int:
        """Load every subdirectory of root_dir as an adapter. Returns count."""
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if not (sub / "meta.json").exists():
                continue
            try:
                self.register(ProceduralAdapter.load(sub))
                count += 1
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return count

    def match(
        self,
        query_hv: np.ndarray,
        threshold: float = 0.3,
        query_text: str = "",
    ) -> list[tuple[float, ProceduralAdapter]]:
        """Return all matching skills sorted by score descending.

        If query_text is provided, skills with matching trigger_patterns
        get a +0.5 boost (clamped to 1.0).
        """
        scored = [(s.match_score(query_hv, query_text), s) for s in self.skills]
        scored = [(score, sk) for score, sk in scored if score >= threshold]
        scored.sort(key=lambda x: -x[0])
        return scored

    def __len__(self) -> int:
        return len(self.skills)
