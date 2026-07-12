"""Download PCFG SET (Hupkes et al. 2020), raw parallel text only.

PCFG SET tests compositional generalization on nested string-edit operations
like `reverse(copy(append(...)))`. 10 primitive operations, nested up to
depth 8.

Source: https://github.com/i-machine-think/am-i-compositional

Writes raw parallel src/tgt files only (no tokenization, no .bin/meta.json):
  data/pcfg/train.src, train.tgt
  data/pcfg/dev.src, dev.tgt
  data/pcfg/test.src, test.tgt

Tokenization happens at load time in the code that consumes these files.

Usage:
  python data/pcfg/download_and_prep.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset"
HERE = Path(__file__).parent.resolve()

FILES = [
    ("train.src", "train.src"),
    ("train.tgt", "train.tgt"),
    ("dev.src", "dev.src"),
    ("dev.tgt", "dev.tgt"),
    ("test.src", "test.src"),
    ("test.tgt", "test.tgt"),
]

# Row counts recorded from the upstream PCFG SET release.
EXPECTED_COUNTS = {
    "test.src": 9721,
}


def _download(rel_url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = f"{BASE_URL}/{rel_url}"
    print(f"Fetching {url} -> {out_path.name}")
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
    HERE.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for rel, out in FILES:
        out_path = HERE / out
        _download(rel, out_path)
        n = _count_lines(out_path)
        expected = EXPECTED_COUNTS.get(out)
        if expected is None:
            print(f"[pcfg/{out}] rows={n} (no expected count recorded)")
            continue
        status = "PASS" if n == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[pcfg/{out}] rows={n} expected={expected} {status}")

    if not all_ok:
        raise SystemExit(1)
    print("pcfg: all checked row counts PASS")


if __name__ == "__main__":
    main()
