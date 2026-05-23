# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Decode the WikiText-2 parquet corpus into a plain UTF-8 .txt file.

The existing pretrain drivers read a flat text file, so this is the one-shot
adapter from the HuggingFace-hosted parquet to RAIN's corpus convention.
Skipped if the output file already exists; the script is re-runnable.

Usage:
    python -m scripts.extract_wikitext2 \\
        --in data/corpora/wikitext2_train.parquet \\
        --out data/corpora/wikitext2_train.txt

Default in/out paths match `scripts/download_corpora.sh`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# pyarrow ships as a dep of `datasets`; if we don't have it, fall back to a
# pandas read which has the same contract.
try:
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except ImportError:  # pragma: no cover -- documented fallback path
    _HAVE_PYARROW = False


def _extract_via_pyarrow(in_path: Path) -> str:
    table = pq.read_table(str(in_path))
    column_name = "text" if "text" in table.column_names else table.column_names[0]
    rows = table.column(column_name).to_pylist()
    return "\n".join(r for r in rows if r and r.strip())


def _extract_via_pandas(in_path: Path) -> str:
    import pandas as pd  # local import to avoid hard dep when pyarrow is present
    df = pd.read_parquet(in_path)
    column_name = "text" if "text" in df.columns else df.columns[0]
    return "\n".join(s for s in df[column_name].astype(str).tolist() if s.strip())


def main() -> int:
    p = argparse.ArgumentParser(description="Extract WikiText-2 parquet -> flat .txt")
    p.add_argument("--in", dest="in_path",
                   default="data/corpora/wikitext2_train.parquet")
    p.add_argument("--out", dest="out_path",
                   default="data/corpora/wikitext2_train.txt")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if the output file already exists.")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.is_file():
        print(f"input parquet not found: {in_path}")
        return 2
    if out_path.is_file() and not args.force:
        size = out_path.stat().st_size
        print(f"output exists already ({size:,} bytes). pass --force to re-extract.")
        return 0

    text = (
        _extract_via_pyarrow(in_path) if _HAVE_PYARROW
        else _extract_via_pandas(in_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}  ({len(text):,} chars, "
          f"{len(set(text)):,} unique chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
