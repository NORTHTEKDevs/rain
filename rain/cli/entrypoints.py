# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Thin entry-point wrappers for pip-installed CLI commands.

Each function corresponds to a `[project.scripts]` entry in pyproject.toml.
The bodies just delegate to the existing scripts in `scripts/` so we
have one place to update logic.

These let users run:
    rain-chat                   (vs python scripts/rain_chat_v2.py)
    rain-demo                   (vs python scripts/rain_net_demo.py)
    rain-bench                  (vs python scripts/benchmark_rain_net.py)
    rain-distill --model llama3.2:3b --n 50
    rain-multimodal
    rain-report
    rain-serve --port 8080
"""

from __future__ import annotations

import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def _run_script(name: str) -> int:
    """Exec a script in the scripts/ directory with sys.argv[1:] forwarded."""
    script = _SCRIPTS_DIR / name
    if not script.exists():
        print(f"rain-net: script not found: {script}", file=sys.stderr)
        return 2
    # Inject the script path as argv[0] and forward the rest.
    sys.argv = [str(script)] + sys.argv[1:]
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    g = {"__name__": "__main__", "__file__": str(script)}
    try:
        exec(code, g)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def chat() -> int:
    """rain-chat: interactive REPL with skills + episodic recall."""
    return _run_script("rain_chat_v2.py")


def demo() -> int:
    """rain-demo: simpler REPL demonstrating the core pipeline."""
    return _run_script("rain_net_demo.py")


def bench() -> int:
    """rain-bench: synthetic 200-fact retrieval vs Jaccard baseline."""
    return _run_script("benchmark_rain_net.py")


def distill() -> int:
    """rain-distill: collect teacher-LLM distillation data via Ollama."""
    return _run_script("run_ollama_distill.py")


def multimodal() -> int:
    """rain-multimodal: text + image + numeric compound retrieval demo."""
    return _run_script("multimodal_demo.py")


def report() -> int:
    """rain-report: run the full v0.1 benchmark suite and emit RESULTS-v0.1.md."""
    return _run_script("rain_net_full_report.py")


def serve() -> int:
    """rain-serve: launch a simple HTTP server exposing POST /query."""
    return _run_script("rain_net_serve.py")
