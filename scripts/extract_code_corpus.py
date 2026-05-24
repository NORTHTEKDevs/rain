# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Build a code-aware Q/A training corpus for RAIN.

Pulls multiple code-Q/A datasets via Hugging Face datasets:
  * sahil2801/CodeAlpaca-20k  -- 20K instruct-style code prompts + solutions
  * openai_humaneval          -- 164 hand-crafted programming problems
  * jondurbin/airoboros-2.2.1  -- mixed code + reasoning (subset)

Renders to the same Q:/A: format as scripts.extract_alpaca so the
result composes cleanly with the existing hybrid_conversational_v1
corpus.

Usage:
    python -m scripts.extract_code_corpus \
        --out data/corpora/code_qa.txt \
        --max 20000 --filter-len 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _format(instruction: str, solution: str) -> str:
    """Same Q/A format as scripts.extract_alpaca."""
    q = instruction.strip()
    a = solution.strip()
    return f"Q: {q}\nA:\n```\n{a}\n```\n"


def _try_load(name: str, split: str = "train"):
    try:
        from datasets import load_dataset

        return load_dataset(name, split=split)
    except Exception as e:
        print(f"  skipping {name}: {e}", file=sys.stderr)
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, default=0, help="cap total rows (0 = no cap)")
    p.add_argument(
        "--filter-len",
        type=int,
        default=3000,
        help="drop examples whose Q+A char length exceeds this (default 3000)",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=["codealpaca", "humaneval"],
        help="which datasets to include",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    with out.open("w", encoding="utf-8") as fh:
        # 1. CodeAlpaca-20k
        if "codealpaca" in args.include:
            print("loading sahil2801/CodeAlpaca-20k ...", file=sys.stderr)
            ds = _try_load("sahil2801/CodeAlpaca-20k", split="train")
            if ds is not None:
                for row in ds:
                    if args.max and kept >= args.max:
                        break
                    instr = row.get("instruction", "")
                    out_text = row.get("output", row.get("response", ""))
                    if not (instr and out_text):
                        continue
                    block = _format(instr, out_text)
                    if len(block) > args.filter_len:
                        dropped += 1
                        continue
                    fh.write(block + "\n")
                    kept += 1

        # 2. OpenAI HumanEval
        if "humaneval" in args.include and (args.max == 0 or kept < args.max):
            print("loading openai_humaneval ...", file=sys.stderr)
            ds = _try_load("openai_humaneval", split="test")
            if ds is not None:
                for row in ds:
                    if args.max and kept >= args.max:
                        break
                    instr = row.get("prompt", "")
                    soln = row.get("canonical_solution", "")
                    if not (instr and soln):
                        continue
                    block = _format(instr, soln)
                    if len(block) > args.filter_len:
                        dropped += 1
                        continue
                    fh.write(block + "\n")
                    kept += 1

    size = out.stat().st_size
    print(
        f"wrote {out}  rows_kept={kept}  rows_dropped={dropped}  bytes={size}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
