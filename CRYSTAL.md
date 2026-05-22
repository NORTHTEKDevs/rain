# RAIN — Project Crystal

> Load this at the start of every session working in this repo.
> Last updated: 2026-05-22 (scaffold creation).

## Stack
- Python 3.11+ (research path), Rust 2024 / MSRV 1.85 (hot kernels via PyO3),
  TypeScript (orchestration + browser via existing cognitive-kernel-polyglot),
  WASM (browser inference), Tauri 2 + React + shadcn (chat UI).
- Tokenizer: SentencePiece BPE 32K.
- Storage: SQLite via sqlx (local); polyglot kernel Postgres adapter (cloud).
- Bipolar 10K-dim VSA state (from Hyperion `vsa_core/`), not the 4K-dim FHRR
  from RCK. RCK code adapts up.
- Build: maturin (Py+Rs), wasm-pack (browser), cargo (Rs workspace), pnpm (TS).

## Key files
- `docs/plans/2026-05-22-rain-design.md` — master design doc. Source of truth
  for architecture decisions.
- `docs/architecture/component-map.md` — which RCK / Hyperion / SOFAR /
  polyglot / Evolve module maps to which RAIN module.
- `docs/architecture/fep-unified-objective.md` — math derivation of the single
  Free Energy objective.
- `docs/architecture/benchmark-suite.md` — Tier 1 (5 novelty), Tier 2 (8 LLM
  parity), Tier 3 (5 architectural soundness) — the v0 acceptance contract.

## Patterns
- **FEP-unified objective.** Every component computes a slice of Free Energy
  or samples from `q`. New components must be derivable in those terms before
  being added.
- **Local-rule learning only at runtime.** Phase 1 bootstrap is the ONLY place
  global gradient descent fires. Phase 2 forever uses PCN, LSM-RLS, Tsetlin
  feedback, FEP rank-1 only.
- **Multi-signal EFE decoder (7 sources).** Adding capability ≈ adding a new
  candidate source, not a new layer.
- **Sourced-over-built.** Before writing new code, check the component map
  to confirm the function isn't already in RCK, Hyperion, SOFAR, polyglot, or
  Evolve.
- **Confidential header on every source file.** Audit script in `scripts/`.
- **No public commits, no benchmarks posted, no marketing material disclosing
  internals.**

## Decisions
- **Binding convention:** Kanerva bipolar 10K-dim (Hyperion). NOT Plate HRR
  (RCK's 4K FHRR). RCK code is adapted up.
- **Calibration source-of-truth:** polyglot kernel `calibration.ts` (Bayesian
  shrinkage, tenant-scoped). NOT RCK's `metacog.py` per-relation tally — RCK's
  tally is consumed BY the polyglot calibration as input, but the
  authoritative store is polyglot.
- **Tokenizer:** SentencePiece BPE 32K at v0 (not deferred to v0.5).
- **Hard out at v0:** aiproof, Rhizome.
- **Chat UI: LLM-equivalent surface.** Looks like ChatGPT/Claude. RAIN
  advantages surface as features (confidence chips, citation toggle, "learned
  X this turn" banners), not as friction.
- **Bootstrap from public pretrained embeddings** (FastText /
  sentence-transformers projected to bipolar). This is NOT an LLM substrate
  in the loop — it is one-shot data init for the codebook, then immediately
  diverge into local-rule training.

## Gotchas
- Hyperion's 10K bipolar primitives and RCK's 4K complex-valued FHRR are not
  bit-compatible. Any code copied from RCK that assumes complex hypervectors
  must be adapted to bipolar sign-space before integration.
- SOFAR's `_w_read`/`_w_write` Conv1D handling is specific to HuggingFace
  Transformers. The transplant onto RAIN's VSA state uses the SVD math only;
  the PyTorch hook plumbing is dropped.
- Polyglot kernel's `llm-gateway.ts` and `local-model.ts` are explicitly NOT
  imported into RAIN. RAIN has no LLM in the inference loop.
- Inline learning per token costs ~5-10ms on CPU. For high-throughput serving,
  toggle background-mode learning via the WorkloadObserver `serving` profile.
- The 18-benchmark Tier 1/2/3 acceptance suite is THE definition of v0 done.
  No PR claims a Tier-1 surface as "complete" without re-running the relevant
  N1-N5 benchmark.

## Next milestone
v0.0.0-scaffold tag → invoke `superpowers:writing-plans` → 12-week
implementation plan → first commits implementing Phase 1.1 (codebook
warm-start from FastText / sentence-transformers).
