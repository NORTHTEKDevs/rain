"""Download COGS (Kim & Linzen 2020), raw TSV only.

COGS tests compositional generalization on natural-language-to-logical-form
parsing. Train (~24K examples) covers a fixed set of constructions; the
generalization split (~21K examples) contains novel compositions of the same
primitives.

Source: https://github.com/najoungkim/COGS

Writes raw TSV files only (no tokenization, no .bin/meta.json):
  data/cogs/raw/train.tsv
  data/cogs/raw/dev.tsv
  data/cogs/raw/test.tsv
  data/cogs/raw/gen.tsv

Each line: <input>\\t<logical_form>\\t<category>. Tokenization happens at
load time in the code that consumes these files.

Usage:
  python data/cogs/download_and_prep.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/najoungkim/COGS/main/data"
HERE = Path(__file__).parent.resolve()

FILES = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
    "gen": "gen.tsv",
}

# Row counts recorded from the upstream COGS release.
EXPECTED_COUNTS = {
    "train": 24155,
    "gen": 21000,
}


def _download(rel_url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{BASE_URL}/{rel_url}"
    print(f"Fetching {url} -> {out_path}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _count_lines(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


def main() -> None:
    raw_dir = HERE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for split, rel in FILES.items():
        out_path = raw_dir / rel
        _download(rel, out_path)
        n = _count_lines(out_path)
        expected = EXPECTED_COUNTS.get(split)
        if expected is None:
            print(f"[cogs/{split}] rows={n} (no expected count recorded)")
            continue
        status = "PASS" if n == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[cogs/{split}] rows={n} expected={expected} {status}")

    if not all_ok:
        raise SystemExit(1)
    print("cogs: all checked row counts PASS")


if __name__ == "__main__":
    main()
