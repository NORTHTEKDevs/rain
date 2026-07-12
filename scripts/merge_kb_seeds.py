"""Merge multiple KB seed JSONLs, dedupe on (subject, relation, object),
optionally augment via WikiText-2 noun-phrase extraction.

Use this to grow the fact pool past 2.4K (currently llama3b_expanded_filtered)
toward 10K-50K facts for v6+ training.

Usage:
    python -m scripts.merge_kb_seeds \
        --in data/kb_seed/llama3b_expanded_filtered.jsonl \
        --in data/kb_seed/llama3b_default.jsonl \
        --in data/kb_seed/qwen30b_default.jsonl \
        --augment-from-corpus data/corpora/wikitext2_train.txt \
        --max-augment 5000 \
        --out data/kb_seed/merged_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = rec.get("subject") or rec.get("s")
            r = rec.get("relation") or rec.get("r")
            o = rec.get("object") or rec.get("o")
            if s and r and o:
                out.append({"subject": str(s), "relation": str(r), "object": str(o)})
    return out


_NPRE = re.compile(r"\b([A-Z][a-z]{3,15})\b")


def _augment_from_corpus(path: Path, max_facts: int) -> list[dict]:
    """Tiny noun-phrase extractor. Looks for capitalized words followed by
    'is' / 'was' / 'are' patterns and creates (subject, isa, object) facts.

    Crude but produces useful KB padding from any text corpus.
    """
    text = path.read_text(encoding="utf-8")
    facts: list[dict] = []
    # Pattern: "Capital Foo is/was/are Capital Bar"
    pat = re.compile(
        r"\b([A-Z][a-zA-Z]{2,20})\s+(?:is|was|are|were|became)\s+(?:a|an|the)?\s*([a-z][a-zA-Z]{2,25})\b"
    )
    seen = set()
    for m in pat.finditer(text):
        s = m.group(1).lower()
        o = m.group(2).lower()
        key = (s, "isa", o)
        if key in seen or s == o:
            continue
        seen.add(key)
        facts.append({"subject": s, "relation": "isa", "object": o})
        if len(facts) >= max_facts:
            break
    return facts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="one or more input JSONLs (use --in multiple times)",
    )
    p.add_argument(
        "--augment-from-corpus",
        default=None,
        help="text corpus to extract additional 'subject isa object' facts from",
    )
    p.add_argument("--max-augment", type=int, default=5000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    all_facts: list[dict] = []
    per_source: list[tuple[str, int]] = []
    for src in args.inputs:
        facts = _read_jsonl(Path(src))
        per_source.append((src, len(facts)))
        all_facts.extend(facts)

    if args.augment_from_corpus:
        aug = _augment_from_corpus(Path(args.augment_from_corpus), args.max_augment)
        per_source.append((f"corpus-extract:{args.augment_from_corpus}", len(aug)))
        all_facts.extend(aug)

    # Dedupe on (subject, relation, object)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for f in all_facts:
        key = (f["subject"].lower(), f["relation"].lower(), f["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for f in deduped:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    print("source                                                  facts")
    for src, n in per_source:
        print(f"  {src:54s}  {n:6d}")
    print(f"  total (with dupes)                                      {len(all_facts):6d}")
    print(f"  deduped -> {out_path}                            {len(deduped):6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
