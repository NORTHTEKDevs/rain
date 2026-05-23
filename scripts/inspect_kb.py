# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Inspect a KB seed JSONL: subject + relation distribution + spot facts.

Useful for debugging KB quality before kicking off a Phase-2 RLAIF run,
deciding whether to re-seed with different topics, or sanity-checking a
filter run's output.

Usage:
    python -m scripts.inspect_kb data/kb_seed/llama3b_expanded.jsonl
    python -m scripts.inspect_kb <path> --top 30 --relation lives_in
    python -m scripts.inspect_kb <path> --subject lion
"""

from __future__ import annotations
import argparse
import collections
import json
import sys
from pathlib import Path


def _read_facts(path: Path) -> list[dict]:
    facts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            facts.append(obj)
    return facts


def _triple(fact: dict) -> tuple[str, str, str] | None:
    s = fact.get("subject") or fact.get("s")
    r = fact.get("relation") or fact.get("r")
    o = fact.get("object") or fact.get("o")
    if not (s and r and o):
        return None
    return (str(s), str(r), str(o))


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a KB seed JSONL")
    p.add_argument("path", help="KB seed JSONL")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--subject", help="filter to facts about this subject")
    p.add_argument("--relation", help="filter to facts on this relation")
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"file not found: {path}", file=sys.stderr)
        return 2
    facts = _read_facts(path)
    print(f"loaded {len(facts)} facts from {path}")

    triples = [_triple(f) for f in facts]
    triples = [t for t in triples if t is not None]
    print(f"valid triples: {len(triples)}")

    subjects = collections.Counter(t[0] for t in triples)
    relations = collections.Counter(t[1] for t in triples)
    objects = collections.Counter(t[2] for t in triples)

    print()
    print(f"unique subjects: {len(subjects)}")
    print(f"unique relations: {len(relations)}")
    print(f"unique objects: {len(objects)}")

    print()
    print(f"top {args.top} subjects (by fact count):")
    for s, n in subjects.most_common(args.top):
        print(f"  {n:4d}  {s}")

    print()
    print(f"top {args.top} relations (by fact count):")
    for r, n in relations.most_common(args.top):
        print(f"  {n:4d}  {r}")

    if args.subject:
        print()
        print(f"facts about subject={args.subject!r}:")
        for t in triples:
            if t[0] == args.subject:
                print(f"  ({t[0]}, {t[1]}, {t[2]})")

    if args.relation:
        print()
        print(f"facts on relation={args.relation!r}:")
        for t in triples:
            if t[1] == args.relation:
                print(f"  ({t[0]}, {t[1]}, {t[2]})")

    # If the file has judge metadata (from filter_seed_with_judge), summarize it.
    judge_facts = [f for f in facts if "judge" in f]
    if judge_facts:
        print()
        print(f"judge-annotated facts: {len(judge_facts)}")
        decisions = collections.Counter(
            f["judge"].get("decision", "n/a") for f in judge_facts
        )
        for k, v in decisions.most_common():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
