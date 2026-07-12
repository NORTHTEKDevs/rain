"""Download SCAN (Lake & Baroni 2018) add-primitive-jump split, raw text only.

SCAN tests systematic compositional generalization. The "add-primitive jump"
split trains on all examples WITHOUT 'jump' and evaluates on all examples
WITH 'jump' -- the canonical compositional-generalization stress test.

Source: https://github.com/brendenlake/SCAN

Writes raw text files only (no tokenization, no .bin/meta.json):
  data/scan/addprim_jump/train.txt
  data/scan/addprim_jump/test.txt

Each line has the form "IN: <command> OUT: <action sequence>". Tokenization
happens at load time in the code that consumes these files (see
experiments/scan_transformer_baseline.py).

Usage:
  python data/scan/download_and_prep.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master"
HERE = Path(__file__).parent.resolve()

FILES = {
    "train": "add_prim_split/tasks_train_addprim_jump.txt",
    "test": "add_prim_split/tasks_test_addprim_jump.txt",
}

# Row counts recorded from the upstream SCAN addprim_jump split.
EXPECTED_COUNTS = {
    "train": 14670,
    "test": 7706,
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
    out_dir = HERE / "addprim_jump"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for split, rel in FILES.items():
        out_path = out_dir / f"{split}.txt"
        _download(rel, out_path)
        n = _count_lines(out_path)
        expected = EXPECTED_COUNTS[split]
        status = "PASS" if n == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[scan/addprim_jump/{split}] rows={n} expected={expected} {status}")

    if not all_ok:
        raise SystemExit(1)
    print("scan/addprim_jump: all row counts PASS")


if __name__ == "__main__":
    main()
