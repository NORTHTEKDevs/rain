# CONFIDENTIAL
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
    print()
    print("Pretrain drivers:")
    print(
        "  python -m scripts.pretrain_hymn       --corpus <path> --out <path>   # numpy CPU reference"
    )
    print(
        "  python -m scripts.pretrain_hymn_torch --corpus <path> --out <path>   # PyTorch + DirectML iGPU"
    )
    print()
    print("KB seed drivers:")
    print(
        "  python -m scripts.seed_kb_from_ollama --model <ollama-name> --out <path>  # distill from local LLM"
    )
    print("  pytest -q -m 'not slow'            -- full 155-test acceptance suite")
    print()
    print("Benchmark drivers:")
    print("  python evals/tier1_novelty/retention.py --out <path>")
    print("  python evals/tier1_novelty/scan.py --out <path>")
    print("  python evals/tier1_novelty/tom.py --out <path>")
    print("  python evals/tier1_novelty/transparency.py --out <path>")
    print("  python evals/tier1_novelty/calibration.py --n 1000 --out <path>")
    print(
        "  python evals/tier2_llm_parity/tiny_shakespeare.py --checkpoint <path> --corpus <path> --out <path>"
    )
    print("  python evals/tier3_soundness/efe_source_mix.py --out <path>")
    print("  python evals/tier3_soundness/nsga2_promotion.py --out <path>")
    print("  python evals/tier3_soundness/continual_stability.py --out <path>")
    print("  python evals/tier3_soundness/dispatch_correctness.py --out <path>")
    print("  python evals/tier3_soundness/sofar_ablation.py --out <path>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
