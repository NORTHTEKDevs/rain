# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN CLI entry point. Currently routes to the demo script."""

from __future__ import annotations
import sys


def main() -> int:
    print("RAIN -- Resonant Active Inference Network")
    print()
    print("This is a private research package. See docs/getting-started.md.")
    print()
    print("Quick demos:")
    print("  python examples/01_chat_e2e.py    -- end-to-end Tier-1 surfaces")
    print("  pytest -q -m 'not slow'            -- full 133-test acceptance suite")
    print()
    print("Benchmark drivers:")
    print("  python evals/tier1_novelty/scan.py --out <path>")
    print("  python evals/tier1_novelty/tom.py --out <path>")
    print("  python evals/tier1_novelty/transparency.py --out <path>")
    print("  python evals/tier1_novelty/calibration.py --n 1000 --out <path>")
    print("  python evals/tier3_soundness/efe_source_mix.py --out <path>")
    print("  python evals/tier3_soundness/nsga2_promotion.py --out <path>")
    print("  python evals/tier3_soundness/continual_stability.py --out <path>")
    print("  python evals/tier3_soundness/dispatch_correctness.py --out <path>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
