# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Concatenate several training corpora with optional oversampling weights.

Use case: blend WikiText-103 (broad factual prose) + KB-derived Q/A pairs
(structured chat format) + Shakespeare (literary register) into a single
HYMN training corpus where each style is seen at a configurable rate.

Usage:
    python -m scripts.compose_corpus \\
        --part data/corpora/wikitext103_train.txt:1 \\
        --part data/corpora/kb_qa.txt:50 \\
        --part data/corpora/tiny_shakespeare.txt:5 \\
        --shuffle-chunks \\
        --out data/corpora/hybrid_v0.txt

Each --part takes 'PATH:WEIGHT'. WEIGHT N concatenates that part N times.
The shuffle is at the line/paragraph level (so we don't tear single
words), not per-character.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Compose a hybrid training corpus")
    p.add_argument("--part", action="append", required=True,
                   help="PATH:WEIGHT (e.g. data/corpora/x.txt:5)")
    p.add_argument("--shuffle-chunks", action="store_true",
                   help="paragraph-shuffle the final corpus (keeps lines intact)")
    p.add_argument("--chunk-on", default="\n\n",
                   help="paragraph separator (default: blank line)")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.rng_seed)
    chunks: list[str] = []
    parts_summary: list[tuple[str, int, int]] = []

    for spec in args.part:
        # Use rpartition so Windows drive colons (e.g. 'C:\foo:5') survive.
        # The WEIGHT is always after the LAST colon.
        path_str, _, weight_str = spec.rpartition(":")
        if not path_str:
            # No colon at all -- treat the whole spec as a path with weight 1.
            path_str = weight_str
            weight_str = "1"
        try:
            weight = max(1, int(weight_str))
        except ValueError:
            # weight_str isn't a number; treat the whole spec as the path.
            path_str = spec
            weight = 1
        path = Path(path_str)
        if not path.is_file():
            print(f"warn: skipping missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        sub_chunks = [c for c in text.split(args.chunk_on) if c.strip()]
        for _ in range(weight):
            chunks.extend(sub_chunks)
        parts_summary.append((str(path), weight, len(sub_chunks)))
        print(f"part: {path.name}  weight={weight}  chunks={len(sub_chunks)}")

    if not chunks:
        print("no chunks composed")
        return 2

    if args.shuffle_chunks:
        rng.shuffle(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(args.chunk_on.join(chunks), encoding="utf-8")
    chars = out_path.stat().st_size
    print(f"wrote {out_path}  ({chars:,} bytes, {len(chunks):,} chunks)")
    print("summary:")
    for path, weight, sub in parts_summary:
        print(f"  {path}: weight={weight} x {sub} chunks = {weight * sub} contributed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
