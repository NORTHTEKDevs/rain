"""json_parser: extract values from JSON via simple path notation.

Patterns:
    "get name from {\"name\": \"Apollo\", \"year\": 1969}"
    "parse {\"a\":[1,2,3]} field a"
    "what is the value of users[0].email in {...}"
"""

from __future__ import annotations

import json
import re


def _find_json(text: str) -> str | None:
    """Find the longest balanced { ... } block in text."""
    # Greedy: find first { and last } and try to parse.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        # Try array.
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
    return text[start : end + 1]


def _dig(obj, path: list[str]):
    """Walk dotted/indexed path into obj."""
    cur = obj
    for part in path:
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                continue
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        return None
    return cur


def _parse_path(text: str) -> list[str] | None:
    """Pull a single field name or dotted path from the query."""
    # Patterns: "field X", "get X", "value of X", "key X"
    m = re.search(
        r"(?:field|get|value of|key)\s+([a-zA-Z_][\w.\[\]]*)", text, re.I
    )
    if m:
        path = m.group(1)
    else:
        # Bare "X from {json}" pattern
        m = re.search(r"\b([a-zA-Z_]\w*)\s+from\s+[\{\[]", text)
        if m:
            path = m.group(1)
        else:
            return None
    # Convert path[i] -> path.i for uniform traversal
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    return [p for p in path.split(".") if p]


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "json_parser: empty query"
    raw = _find_json(query_text)
    if raw is None:
        return "json_parser: no JSON block found in query"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"json_parser: invalid JSON ({e.msg})"
    path = _parse_path(query_text)
    if path is None:
        # No path -> just pretty-print top-level keys/length
        if isinstance(obj, dict):
            return f"keys: {sorted(obj.keys())}"
        if isinstance(obj, list):
            return f"list of length {len(obj)}"
        return f"value: {obj!r}"
    val = _dig(obj, path)
    if val is None:
        return f"json_parser: path {'.'.join(path)} not found"
    return json.dumps(val) if not isinstance(val, str) else val
