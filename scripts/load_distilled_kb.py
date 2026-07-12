"""Load all distillation .jsonl files in a directory into a single
RainNet KB. Then optionally launch an interactive REPL or run a
self-eval over the queries.

Run:
    python scripts/load_distilled_kb.py                  # load + REPL
    python scripts/load_distilled_kb.py --eval           # load + self-eval
    python scripts/load_distilled_kb.py --eval --out FILE  # save metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rain.core.rain_net import RainNet, RainNetConfig
from rain.training.distillation import DistillationPipeline


def load_dir(net: RainNet, distill_dir: Path) -> int:
    """Ingest every *.jsonl in distill_dir as facts. Returns count."""
    count = 0
    for path in distill_dir.glob("*.jsonl"):
        for ex in DistillationPipeline.load(path):
            fact_text = f"Q: {ex.query} A: {ex.teacher_answer}"
            net.ingest_fact(
                text=fact_text, source=f"distill::{ex.teacher_backend}::{path.stem}"
            )
            count += 1
    return count


def self_eval(net: RainNet, distill_dir: Path) -> dict:
    """For each (query, answer) in the distill files, check if the
    query retrieves the matching fact at top-1/top-3/top-5.

    The matching fact's text contains 'Q: <query>' so we can detect
    a correct retrieval by string match against the cited fact text.
    """
    correct_at = {1: 0, 3: 0, 5: 0}
    total = 0
    for path in distill_dir.glob("*.jsonl"):
        for ex in DistillationPipeline.load(path):
            report = net.answer(ex.query)
            cited_ids = [f.fact_id for f in report.cited_facts]
            cited_texts = [f.text for f in report.cited_facts]
            # Check if any cited fact contains the query (round-trip check).
            for k in (1, 3, 5):
                if any(ex.query[:60] in t for t in cited_texts[:k]):
                    correct_at[k] += 1
            total += 1
    return {
        "total_queries": total,
        "top1": correct_at[1] / total if total else 0.0,
        "top3": correct_at[3] / total if total else 0.0,
        "top5": correct_at[5] / total if total else 0.0,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir", default="data/distill", help="directory containing *.jsonl files"
    )
    p.add_argument("--dim", type=int, default=10_000)
    p.add_argument("--eval", action="store_true", help="run round-trip self-eval")
    p.add_argument("--out", default=None, help="if --eval, write metrics JSON here")
    p.add_argument("--n-candidates", type=int, default=2)
    args = p.parse_args(argv)

    print(f"=== RAIN-Net distilled KB loader ===")
    distill_dir = Path(args.dir)
    if not distill_dir.exists():
        print(f"ERROR: directory not found: {distill_dir}")
        return 1

    files = sorted(distill_dir.glob("*.jsonl"))
    print(f"Found {len(files)} .jsonl files in {distill_dir}:")
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        print(f"  - {f.name} ({n} examples)")
    print()

    net = RainNet(
        config=RainNetConfig(
            dim=args.dim,
            n_candidates=args.n_candidates,
            semantic_top_k=5,
        )
    )

    print("Loading into RainNet semantic memory...")
    n = load_dir(net, distill_dir)
    print(f"Loaded {n} distilled examples. KB size: {len(net.memory.semantic)}.")
    print()

    if args.eval:
        print("Running self-eval (round-trip retrieval over loaded queries)...")
        metrics = self_eval(net, distill_dir)
        print(f"  total queries: {metrics['total_queries']}")
        print(f"  top-1 retrieval: {metrics['top1']:.3f}")
        print(f"  top-3 retrieval: {metrics['top3']:.3f}")
        print(f"  top-5 retrieval: {metrics['top5']:.3f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"  saved metrics to {args.out}")
        return 0

    # Interactive REPL.
    print("Type 'exit' to quit, or any question to query the KB.")
    print()
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q or q == "exit":
            return 0
        report = net.answer(q)
        print(report.human_format())
        print()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
