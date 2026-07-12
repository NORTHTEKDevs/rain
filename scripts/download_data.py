"""Download all raw datasets needed to reproduce the RAIN-CG benchmark suite.

Runs the three per-dataset downloaders in sequence:
  data/scan/download_and_prep.py  (SCAN addprim_jump)
  data/cogs/download_and_prep.py  (COGS)
  data/pcfg/download_and_prep.py  (PCFG SET)

Each downloader prints PASS/FAIL row-count verification and exits non-zero
on a mismatch; this script propagates the first failure.

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DOWNLOADERS = [
    REPO / "data" / "scan" / "download_and_prep.py",
    REPO / "data" / "cogs" / "download_and_prep.py",
    REPO / "data" / "pcfg" / "download_and_prep.py",
]


def main() -> None:
    for script in DOWNLOADERS:
        print(f"=== {script.relative_to(REPO)} ===")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code:
                print(f"FAILED: {script.relative_to(REPO)}")
                raise
    print("\nAll datasets downloaded and verified.")


if __name__ == "__main__":
    main()
