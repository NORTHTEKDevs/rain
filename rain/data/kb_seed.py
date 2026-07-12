"""Seed a ShardedKB from a JSONL facts file.

Each line must be a JSON object carrying a subject, relation, object triple.
Two key naming conventions are accepted:
  * Short: {"s": ..., "r": ..., "o": ...}  -- original RAIN format.
  * Long : {"subject": ..., "relation": ..., "object": ...} -- produced by
    `scripts/seed_kb_from_ollama.py`. Extra keys (e.g., topic, source)
    are tolerated and silently ignored.

Library-canonical location. scripts/seed_kb.py re-exports from here
for backwards compatibility.
"""

from __future__ import annotations

import json

from rain.core.knowledge_base import ShardedKB


def _extract_triple(fact: dict) -> tuple[str, str, str] | None:
    """Pull (s, r, o) out of either the short or long-form fact dict."""
    if "s" in fact and "r" in fact and "o" in fact:
        return str(fact["s"]), str(fact["r"]), str(fact["o"])
    if "subject" in fact and "relation" in fact and "object" in fact:
        return str(fact["subject"]), str(fact["relation"]), str(fact["object"])
    return None


def seed_from_jsonl(kb: ShardedKB, path: str) -> int:
    """Load every triple in `path` into `kb`. Returns number of facts written.

    Lines that don't carry a complete triple (either schema) are silently
    skipped -- the Ollama distillation pipeline already runs its own
    schema validation upstream, so a malformed line at this point is a
    bug in that pipeline, not a per-call concern.
    """
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fact = json.loads(line)
            triple = _extract_triple(fact)
            if triple is None:
                continue
            kb.write(*triple)
            n += 1
    return n
