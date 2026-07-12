"""Pull tatsu-lab/alpaca and render to char-level Q/A training corpus.

The Alpaca dataset is a 52K instruction/response dataset distilled from
text-davinci-003. We use it ONLY for character-level pre-training of
HYMN so the sequence-engine learns dialogue surface form -- not for
instruction tuning in the LLM sense.

Output format (one example per blank-line-separated block):

    Q: <instruction>[ + input if present]
    A: <response>

Matches the format scripts/kb_to_qa_corpus.py emits, so a composer can
mix Alpaca + KB Q/A in one training run.

CLI:

    python -m scripts.extract_alpaca \
        --out data/corpora/alpaca_qa.txt \
        [--max 50000] [--filter-len 1500]

Notes:
- The default dataset is tatsu-lab/alpaca (52,002 rows, ~22 MB JSONL).
- We drop examples whose total Q+A length exceeds --filter-len chars
  because very long responses dominate gradient updates and the HYMN
  context window is small.
- License: Alpaca's instructions are CC BY 4.0; the model outputs are
  subject to the OpenAI terms of service. Not used for commercial
  redistribution; only as RAIN training corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _format_one(instruction: str, input_text: str, output: str) -> str:
    q = instruction.strip()
    if input_text and input_text.strip():
        q = q + "\n" + input_text.strip()
    a = output.strip()
    return f"Q: {q}\nA: {a}\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Extract Alpaca to RAIN char-level Q/A corpus")
    p.add_argument(
        "--dataset",
        default="tatsu-lab/alpaca",
        help="HF dataset id; default tatsu-lab/alpaca (52K rows)",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True, help="output text file")
    p.add_argument(
        "--max",
        type=int,
        default=0,
        help="cap row count (0 = no cap)",
    )
    p.add_argument(
        "--filter-len",
        type=int,
        default=1500,
        help="drop examples whose Q+A char length exceeds this; "
        "default 1500 (HYMN context windows are small)",
    )
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: pip install datasets", file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.dataset} [{args.split}] ...", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if args.max and kept >= args.max:
                break
            block = _format_one(
                row.get("instruction", ""),
                row.get("input", ""),
                row.get("output", ""),
            )
            if len(block) > args.filter_len:
                dropped += 1
                continue
            fh.write(block)
            fh.write("\n")
            kept += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1} rows scanned, {kept} kept", file=sys.stderr)

    size = out_path.stat().st_size
    print(
        f"wrote {out_path}  rows_kept={kept}  rows_dropped={dropped}  bytes={size}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
