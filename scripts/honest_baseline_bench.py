"""Honest competitive benchmark: RAIN-Net retrieval vs production
sentence-transformer + cosine vs naive Jaccard, on the SAME corpus.

Production RAG today uses a learned encoder (all-MiniLM-L6-v2 is the
standard cheap one) plus a vector index. This script measures whether
RAIN-Net's HV n-gram encoder beats, matches, or loses to that baseline
on the same 500-fact distilled KB we built today.

This is the benchmark that matters for "are we beating competition."

Run:
    python scripts/honest_baseline_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from rain.core.rain_net import RainNet, RainNetConfig
from rain.training.distillation import DistillationPipeline


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    print("=== Honest competitive retrieval benchmark ===\n")

    distill_dir = Path("data/distill")
    pairs: list[tuple[str, str, str]] = []  # (fact_text, query, fact_id)
    for path in sorted(distill_dir.glob("*.jsonl")):
        for ex in DistillationPipeline.load(path):
            fact_text = f"Q: {ex.query} A: {ex.teacher_answer}"
            fid = f"{path.stem}::{len(pairs)}"
            pairs.append((fact_text, ex.query, fid))
    print(f"Total pairs (fact, query, id): {len(pairs)}\n")
    if len(pairs) < 100:
        print("Not enough pairs to benchmark; need >=100. Run distillation first.")
        return 1

    fact_texts = [p[0] for p in pairs]
    queries = [p[1] for p in pairs]
    ids = [p[2] for p in pairs]

    # --- 1. Jaccard lexical baseline ---
    print("--- 1. Jaccard lexical baseline ---")
    t0 = time.time()
    correct_at = {1: 0, 3: 0, 5: 0}
    for q, fid_true in zip(queries, ids):
        scored = [(jaccard(q, ft), ids[i]) for i, ft in enumerate(fact_texts)]
        scored.sort(key=lambda x: -x[0])
        top5 = [fid for _, fid in scored[:5]]
        for k in (1, 3, 5):
            if fid_true in top5[:k]:
                correct_at[k] += 1
    elapsed = time.time() - t0
    jacc = {k: correct_at[k] / len(queries) for k in (1, 3, 5)}
    print(f"  top-1={jacc[1]:.3f}  top-3={jacc[3]:.3f}  top-5={jacc[5]:.3f}")
    print(f"  time: {elapsed:.1f}s ({elapsed*1000/len(queries):.1f}ms/query)\n")

    # --- 2. Sentence-transformer (all-MiniLM-L6-v2) ---
    print("--- 2. Sentence-transformer (all-MiniLM-L6-v2) ---")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  model loaded; encoding {len(fact_texts)} facts...")
    t0 = time.time()
    fact_emb = st_model.encode(fact_texts, show_progress_bar=False, convert_to_numpy=True)
    fact_emb = fact_emb / (np.linalg.norm(fact_emb, axis=1, keepdims=True) + 1e-9)
    print(f"  facts encoded in {time.time()-t0:.1f}s")
    print(f"  encoding {len(queries)} queries...")
    t0 = time.time()
    q_emb = st_model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    encode_time = time.time() - t0
    print(f"  queries encoded in {encode_time:.1f}s")
    t0 = time.time()
    sims = q_emb @ fact_emb.T  # (N_q, N_f)
    correct_at = {1: 0, 3: 0, 5: 0}
    for i, fid_true in enumerate(ids):
        top5_idx = np.argsort(-sims[i])[:5]
        top5_ids = [ids[j] for j in top5_idx]
        for k in (1, 3, 5):
            if fid_true in top5_ids[:k]:
                correct_at[k] += 1
    elapsed = time.time() - t0
    st = {k: correct_at[k] / len(queries) for k in (1, 3, 5)}
    print(f"  top-1={st[1]:.3f}  top-3={st[3]:.3f}  top-5={st[5]:.3f}")
    print(f"  retrieval time: {elapsed:.2f}s ({elapsed*1000/len(queries):.1f}ms/query)\n")

    # --- 3. RAIN-Net (n-gram HV) ---
    print("--- 3. RAIN-Net (n-gram HV, our current impl) ---")
    net = RainNet(config=RainNetConfig(dim=10_000, n_candidates=1, semantic_top_k=5))
    for fact_text, _q, fid in pairs:
        net.ingest_fact(fact_text, fact_id=fid)
    print(f"  {len(pairs)} facts ingested into HV memory")
    t0 = time.time()
    correct_at = {1: 0, 3: 0, 5: 0}
    for q, fid_true in zip(queries, ids):
        results = net.memory.semantic.search(net.encoder_bank.encode("text", q), top_k=5)
        top5 = [f.fact_id for _s, f in results]
        for k in (1, 3, 5):
            if fid_true in top5[:k]:
                correct_at[k] += 1
    elapsed = time.time() - t0
    rn = {k: correct_at[k] / len(queries) for k in (1, 3, 5)}
    print(f"  top-1={rn[1]:.3f}  top-3={rn[3]:.3f}  top-5={rn[5]:.3f}")
    print(f"  retrieval time: {elapsed:.2f}s ({elapsed*1000/len(queries):.1f}ms/query)\n")

    # --- 4. Hybrid: bind sentence-transformer embedding INTO HV space ---
    # This is the upgrade path: keep RAIN-Net's HV substrate, but use a
    # learned encoder to produce semantically-meaningful HVs.
    print("--- 4. HYBRID: sentence-transformer -> HV (bipolar quantized) ---")
    # Project 384-d MiniLM embedding into 10000-d HV via random projection.
    rng = np.random.default_rng(0)
    P = rng.standard_normal((fact_emb.shape[1], 10_000)).astype(np.float32) / np.sqrt(10_000)
    hv_facts = np.sign(fact_emb @ P).astype(np.float32)  # bipolar HVs
    hv_facts = np.where(hv_facts == 0.0, 1.0, hv_facts)
    hv_queries = np.sign(q_emb @ P).astype(np.float32)
    hv_queries = np.where(hv_queries == 0.0, 1.0, hv_queries)
    # Normalise for cosine
    hv_f = hv_facts / (np.linalg.norm(hv_facts, axis=1, keepdims=True) + 1e-9)
    hv_q = hv_queries / (np.linalg.norm(hv_queries, axis=1, keepdims=True) + 1e-9)
    t0 = time.time()
    sims = hv_q @ hv_f.T
    correct_at = {1: 0, 3: 0, 5: 0}
    for i, fid_true in enumerate(ids):
        top5_idx = np.argsort(-sims[i])[:5]
        top5_ids = [ids[j] for j in top5_idx]
        for k in (1, 3, 5):
            if fid_true in top5_ids[:k]:
                correct_at[k] += 1
    elapsed = time.time() - t0
    hy = {k: correct_at[k] / len(queries) for k in (1, 3, 5)}
    print(f"  top-1={hy[1]:.3f}  top-3={hy[3]:.3f}  top-5={hy[5]:.3f}")
    print(f"  retrieval time: {elapsed:.2f}s ({elapsed*1000/len(queries):.1f}ms/query)\n")

    # --- Summary ---
    print("=" * 60)
    print(f"{'Method':<40} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    print("-" * 60)
    print(f"{'Jaccard (lexical)':<40} {jacc[1]:>7.3f} {jacc[3]:>7.3f} {jacc[5]:>7.3f}")
    print(f"{'sentence-transformer (production baseline)':<40} {st[1]:>7.3f} {st[3]:>7.3f} {st[5]:>7.3f}")
    print(f"{'RAIN-Net n-gram HV (current)':<40} {rn[1]:>7.3f} {rn[3]:>7.3f} {rn[5]:>7.3f}")
    print(f"{'HYBRID: ST embedding -> HV':<40} {hy[1]:>7.3f} {hy[3]:>7.3f} {hy[5]:>7.3f}")
    print()

    # Verdict
    print("--- Verdict ---")
    win_v_jacc = rn[1] > jacc[1]
    win_v_st = rn[1] > st[1]
    print(f"  Beats Jaccard:                {'YES' if win_v_jacc else 'NO'}  ({rn[1]:.3f} vs {jacc[1]:.3f})")
    print(f"  Beats sentence-transformer:   {'YES' if win_v_st else 'NO'}  ({rn[1]:.3f} vs {st[1]:.3f})")
    if not win_v_st:
        gap = st[1] - rn[1]
        print(f"  -> RAIN-Net loses to ST by {gap*100:.1f} pts at top-1")
    hybrid_beats_st = hy[1] > st[1]
    hybrid_beats_rn = hy[1] > rn[1]
    print(f"  Hybrid beats ST:              {'YES' if hybrid_beats_st else 'NO'}  ({hy[1]:.3f} vs {st[1]:.3f})")
    print(f"  Hybrid beats RAIN-Net:        {'YES' if hybrid_beats_rn else 'NO'}  ({hy[1]:.3f} vs {rn[1]:.3f})")

    # Save raw numbers
    import json
    out = {
        "n_pairs": len(pairs),
        "methods": {
            "jaccard": jacc,
            "sentence_transformer": st,
            "rain_net_ngram_hv": rn,
            "hybrid_st_to_hv": hy,
        },
    }
    Path("docs/HONEST_BASELINE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("Raw metrics: docs/HONEST_BASELINE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
