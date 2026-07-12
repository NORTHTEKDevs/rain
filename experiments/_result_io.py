"""Shared result-JSON persistence helper for experiments/ scripts.

Writes files conforming to the raincg/results/<bench>__<system>__seed<seed>.json
contract. Tries to reuse raincg.bench.common.wilson_ci if it exists; falls back
to a local implementation (unit A may not have landed it yet).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

try:
    from raincg.bench.common import wilson_ci  # type: ignore
except ImportError:
    def wilson_ci(correct: int, n: int) -> tuple[float, float]:
        """Wilson score interval, 95% confidence."""
        if n == 0:
            return (0.0, 1.0)
        z = 1.959963984540054
        p = correct / n
        denom = 1 + z * z / n
        center = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lo = (center - margin) / denom
        hi = (center + margin) / denom
        return (max(0.0, lo), min(1.0, hi))


def emit_result(path: str | Path, **fields) -> dict:
    """Build and write a contract-shaped result JSON. Returns the dict written.

    Required fields (per SHARED CONTRACT): bench, system, split, n, correct, seed.
    accuracy/ci95/timestamp/evidence_tier are filled in if not given explicitly.
    """
    n = fields.get("n")
    correct = fields.get("correct")
    if "accuracy" not in fields and n is not None and correct is not None:
        fields["accuracy"] = correct / n if n else 0.0
    if "ci95" not in fields and n is not None and correct is not None:
        fields["ci95"] = list(wilson_ci(correct, n))
    fields.setdefault("params", 0)
    fields.setdefault("train_s", 0.0)
    fields.setdefault("eval_s", 0.0)
    fields.setdefault("config", {})
    fields.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    fields.setdefault("evidence_tier", "measured-fresh")
    fields.setdefault("notes", "")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)
    return fields
