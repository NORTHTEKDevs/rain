# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Decode WikiText-103 parquet shards to a flat .txt corpus.

Used by Phase-1 v0.5 pretrain. Same pattern as scripts/extract_wikitext2.py
but takes a directory of shard files and concatenates them.

Usage:
    python -m scripts.extract_wikitext103 \\
        --in-dir data/corpora \\
        --pattern 'wikitext103_train_*.parquet' \\
        --out data/corpora/wikitext103_train.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data/corpora")
    p.add_argument("--pattern", default="wikitext103_train_*.parquet")
    p.add_argument("--out", default="data/corpora/wikitext103_train.txt")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    paths = sorted(in_dir.glob(args.pattern))
    if not paths:
        print(f"no shards matching {args.pattern} under {in_dir}")
        return 2
    out_path = Path(args.out)
    if out_path.is_file() and not args.force:
        print(f"output exists: {out_path} ({out_path.stat().st_size:,} bytes). "
              f"pass --force to re-extract.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for shard in paths:
            print(f"reading {shard.name} ({shard.stat().st_size:,} bytes)")
            table = pq.read_table(str(shard))
            column = "text" if "text" in table.column_names else table.column_names[0]
            rows = table.column(column).to_pylist()
            for row in rows:
                if row and row.strip():
                    fh.write(row)
                    fh.write("\n")
                    total += 1
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, {total:,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
