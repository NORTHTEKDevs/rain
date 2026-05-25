# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Math solver procedural skill: safe arithmetic evaluation.

Loaded by rain.core.procedural_adapter.ProceduralAdapter.load() when a
RainNet routes a math-style query to this skill.

The handler signature is enforced:
    def invoke(query_text: str, kb) -> str

Returns an answer string. On parse failure returns a clear "could not
parse" message rather than raising -- skills must degrade gracefully.
"""

from __future__ import annotations

import re


_ARITH_CHARS = re.compile(r"^[\s\d+\-*/().,%eE]+$")


def _safe_eval(expr: str) -> float | None:
    """Evaluate a constrained arithmetic expression with no builtins."""
    cleaned = expr.strip().replace(",", "")
    if not cleaned or not _ARITH_CHARS.match(cleaned):
        return None
    # Replace ^ with ** for python pow if present.
    py_expr = cleaned.replace("^", "**")
    try:
        return float(eval(py_expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _extract_arithmetic(text: str) -> str | None:
    """Find the longest arithmetic substring in the query.

    Match must START with a digit, paren, or unary sign so we don't
    swallow trailing letters like the 'e' of 'compute' or 'evaluate'.
    """
    matches = re.findall(
        r"[\d(\-+][\d+\-*/().,%eE.\s]{2,}", text
    )
    matches = [m.strip() for m in matches if re.search(r"\d", m)]
    if not matches:
        return None
    return max(matches, key=len)


def invoke(query_text: str, kb) -> str:
    """Public skill entrypoint. Returns an answer or graceful failure."""
    if not query_text:
        return "math_solver: empty query"
    expr = _extract_arithmetic(query_text)
    if expr is None:
        return f"math_solver: could not parse arithmetic from {query_text!r}"
    val = _safe_eval(expr)
    if val is None:
        return f"math_solver: could not evaluate {expr!r}"
    # Format integer-valued floats as ints for human readability.
    if val.is_integer():
        return f"{int(val)}"
    return f"{val:.6g}"
