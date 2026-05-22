# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Seed a ShardedKB from a JSONL facts file.

Each line must be a JSON object with keys: s (subject), r (relation), o (object).

Library-canonical location. scripts/seed_kb.py re-exports from here
for backwards compatibility.
"""

from __future__ import annotations
import json
from rain.core.knowledge_base import ShardedKB


def seed_from_jsonl(kb: ShardedKB, path: str) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fact = json.loads(line)
            kb.write(fact["s"], fact["r"], fact["o"])
            n += 1
    return n
