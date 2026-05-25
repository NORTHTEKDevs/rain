# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Bulk Ollama distillation runner.

Iterates through every domain in _CURRICULUM_TOPICS, asks the teacher
N queries per domain, persists each domain to its own JSONL file in
data/distill/v1_<domain>.jsonl. Resumable: skips already-completed
domains (presence + line count check).

Run:
    python scripts/bulk_distill.py --model llama3.2:3b --per-domain 150
    python scripts/bulk_distill.py --domains science,history,art --per-domain 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rain.training.distillation import (
    DistillationPipeline,
    OllamaTeacher,
    synthetic_curriculum_queries,
)
from rain.training.distillation import _CURRICULUM_TOPICS as ALL_DOMAINS


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--per-domain", type=int, default=150)
    p.add_argument(
        "--domains",
        default=None,
        help="comma-separated subset of domains; default = all 16",
    )
    p.add_argument("--out-dir", default="data/distill")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip domains where output JSONL already has >=per-domain lines",
    )
    args = p.parse_args(argv)

    domain_list = (
        [d.strip() for d in args.domains.split(",")] if args.domains else list(ALL_DOMAINS.keys())
    )
    print(f"=== Bulk distillation across {len(domain_list)} domains ===")
    print(f"  model:      {args.model}")
    print(f"  per-domain: {args.per_domain}")
    print(f"  total tgt:  {len(domain_list) * args.per_domain}")
    print(f"  out-dir:    {args.out_dir}")
    print()

    teacher = OllamaTeacher(model=args.model, host=args.host)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    overall_facts = 0
    overall_tokens = 0
    for di, domain in enumerate(domain_list, 1):
        out_path = Path(args.out_dir) / f"v1_{domain}.jsonl"
        if args.resume and out_path.exists():
            existing = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
            if existing >= args.per_domain:
                print(f"[{di}/{len(domain_list)}] {domain}: SKIP ({existing} already collected)")
                overall_facts += existing
                continue
        if domain not in ALL_DOMAINS:
            print(f"[{di}/{len(domain_list)}] {domain}: UNKNOWN, skipping")
            continue

        # Fresh pipeline so .examples doesn't pile across domains.
        pipe = DistillationPipeline(teacher=teacher)
        queries = synthetic_curriculum_queries(domain, n=args.per_domain, seed=di)
        print(f"[{di}/{len(domain_list)}] {domain}: collecting {len(queries)} examples...")
        t0 = time.time()
        n_ok = 0
        n_fail = 0
        for qi, q in enumerate(queries, 1):
            ex = pipe.collect_example(q, metadata={"domain": domain})
            if ex is None:
                n_fail += 1
                continue
            n_ok += 1
            if qi % 25 == 0:
                elapsed = time.time() - t0
                rate = n_ok / elapsed if elapsed > 0 else 0
                print(
                    f"  [{di}/{len(domain_list)}] {domain}: {n_ok}/{len(queries)} "
                    f"({rate:.2f} ex/s, {elapsed:.0f}s elapsed)"
                )
        # Persist this domain
        pipe.save(out_path)
        elapsed = time.time() - t0
        overall_facts += n_ok
        overall_tokens += pipe.token_budget_used
        total_elapsed = time.time() - overall_start
        print(
            f"  -> {domain}: {n_ok} ok / {n_fail} fail in {elapsed:.0f}s "
            f"(tokens={pipe.token_budget_used}); total_so_far={overall_facts} facts "
            f"in {total_elapsed:.0f}s"
        )

    final_elapsed = time.time() - overall_start
    print()
    print("=== Bulk distillation complete ===")
    print(f"  facts collected: {overall_facts}")
    print(f"  total tokens:    {overall_tokens}")
    print(f"  wall time:       {final_elapsed:.0f}s")
    print(f"  cost:            $0 (local Ollama)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
