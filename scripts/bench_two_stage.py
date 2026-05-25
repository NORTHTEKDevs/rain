# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Two-stage retrieval benchmark: speed + quality vs single-stage RAIN-Net."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.two_stage_retrieve import TwoStageRetriever
from rain.training.distillation import DistillationPipeline


def main() -> int:
    print("=== Two-stage retrieval bench ===\n")
    distill_dir = Path("data/distill")
    pairs: list[tuple[str, str, str]] = []
    for path in sorted(distill_dir.glob("*.jsonl")):
        for ex in DistillationPipeline.load(path):
            ftext = f"Q: {ex.query} A: {ex.teacher_answer}"
            fid = f"{path.stem}::{len(pairs)}"
            pairs.append((ftext, ex.query, fid))
    print(f"Total pairs: {len(pairs)}")
    if len(pairs) < 100:
        return 1

    fact_texts = [p[0] for p in pairs]
    queries = [p[1] for p in pairs]
    fids = [p[2] for p in pairs]

    # Single-stage RAIN-Net baseline (from honest benchmark).
    print("\n--- Single-stage RAIN-Net (slow path) ---")
    net = RainNet(config=RainNetConfig(dim=10_000, n_candidates=1, semantic_top_k=5))
    for t, fid in zip(fact_texts, fids, strict=True):
        net.ingest_fact(t, fact_id=fid)
    t0 = time.time()
    correct = {1: 0, 3: 0, 5: 0}
    for q, fid_true in zip(queries, fids, strict=True):
        r = net.memory.semantic.search(net.encoder_bank.encode("text", q), top_k=5)
        top5 = [f.fact_id for _s, f in r]
        for k in (1, 3, 5):
            if fid_true in top5[:k]:
                correct[k] += 1
    elapsed = time.time() - t0
    print(f"  top-1={correct[1]/len(queries):.3f} top-3={correct[3]/len(queries):.3f} top-5={correct[5]/len(queries):.3f}")
    print(f"  time: {elapsed:.1f}s ({elapsed*1000/len(queries):.1f}ms/query)")

    # Two-stage retriever.
    print("\n--- Two-stage retrieval (shortlist=20 -> rerank=5) ---")
    net2 = RainNet(config=RainNetConfig(dim=10_000, n_candidates=1, semantic_top_k=5))
    retriever = TwoStageRetriever(net=net2)
    print(f"  bulk-ingesting {len(pairs)} facts...")
    t0 = time.time()
    retriever.bulk_ingest(fact_texts, ["test"] * len(fact_texts), fact_ids=list(fids))
    print(f"  ingest: {time.time()-t0:.1f}s")

    t0 = time.time()
    correct = {1: 0, 3: 0, 5: 0}
    shortlist_ms = []
    rerank_ms = []
    for q, fid_true in zip(queries, fids, strict=True):
        results = retriever.retrieve(q, shortlist_k=20, final_k=5)
        top5 = []
        for _s, f in results:
            top5.append(f.fact_id)
        for k in (1, 3, 5):
            if fid_true in top5[:k]:
                correct[k] += 1
        if retriever._last_stats:
            shortlist_ms.append(retriever._last_stats.shortlist_ms)
            rerank_ms.append(retriever._last_stats.rerank_ms)
    elapsed = time.time() - t0
    print(f"  top-1={correct[1]/len(queries):.3f} top-3={correct[3]/len(queries):.3f} top-5={correct[5]/len(queries):.3f}")
    print(f"  total: {elapsed:.1f}s ({elapsed*1000/len(queries):.1f}ms/query)")
    if shortlist_ms:
        print(f"  shortlist mean: {np.mean(shortlist_ms):.1f}ms")
        print(f"  rerank    mean: {np.mean(rerank_ms):.1f}ms")

    print("\n--- Speed-up ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
