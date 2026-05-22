# RAIN — Resonant Active Inference Network

**Design document, v0**
**Author:** Kristian Baer / NORTHTEKDevs
**Date:** 2026-05-22
**Status:** Draft, brainstorm-approved, pre-implementation
**Classification:** CONFIDENTIAL

---

## 0. One-sentence claim

A bipolar 10K-dim VSA state evolved by a HYMN MLP update rule, routed by SOFAR frequency-banded SVD beam-steering, decoded by a multi-signal Expected Free Energy decoder over seven candidate sources, learned by local rules only (PCN + LSM-RLS + Tsetlin + FEP-rank-1), grounded by a sharded HRR knowledge base with per-relation Bayesian calibration, organized by a structural self-model + theory-of-mind + metacognition, evolved at the population level by NSGA-II + champion/challenger, dispatched across WASM/Rust/Go/TS for inference, and unified by a single Free Energy objective.

That sentence is the model. Every term maps to existing code in one of five prior NORTHTEKDevs / Frostbyte repos.

---

## 1. Architecture overview and positioning

RAIN is a new class of generative AI that is *not* an LLM and *not* a state-space model. The whole system is one Active Inference loop (FEP unifying frame) implemented over a Vector Symbolic substrate. No global backprop in steady state. No attention. No transformer block. No pretrained LM in the loop at inference.

### Position in the 2026 landscape

| Class | Examples | RAIN's position |
|---|---|---|
| Frontier LLMs | GPT-4, Claude, Gemini | NOT this. Different category. |
| Hybrid SSM/attention | RWKV-7, LFM2, Mamba-3, RecurrentGemma | The realistic peer set for benchmarking. |
| Pure SSM/RWKV | Mamba-1/2, RWKV-6 | Adjacent in spirit, different mechanism. |
| Neuro-symbolic | KG+LLM, MindMap, Decoding-on-Graphs | Symbolic layer is retrieval; RAIN's is per-token calibration. |
| Predictive coding / FEP LMs | Whittington PC, AIxIA-2024 FEP | Same family. They top out small. RAIN composes them with VSA + EFE decoder. |

### Four defensible novelty claims (prior-art verified, 2026-05-21 research pass)

1. **VSA-as-primary-generative-substrate.** No published work uses HRR/Plate/Kanerva binding as the primary substrate for autoregressive text generation. Closest published work: VSA used to *probe* LLMs (arXiv:2509.25045) or encode Lisp (arXiv:2511.08767). Unoccupied territory.
2. **Structural self-model + theory-of-mind as first-class state objects.** All published ToM/self-model work uses prompting or external BDI modules over frozen LLMs. RCK already bakes them into VSA state. This is the strongest novelty claim per the research.
3. **Multi-objective NSGA-II Pareto across polyglot runtimes + Bayesian champion/challenger promotion.** Closest analog (EOE 2025, EvoMoE 2025) does evolutionary ops on experts within a single runtime. WASM/Rust/Go/TS Pareto + beta-binomial promotion has no precedent.
4. **Continual learning by construction, not by mitigation.** Local-rule learning (PCN + LSM-RLS + Tsetlin + FEP rank-1) has no global weight overwrite → no catastrophic forgetting by design. Published continual-learning survey (CSUR 2025) shows this is unsolved for gradient-based LMs.

Two weaker but defensible claims:

5. **Per-relation Bayesian calibration as decoding-time signal** (not retrieval-time).
6. **Active Inference as the unifying training objective** for every component.

### Honest weaknesses

| Concern | Source | Design response |
|---|---|---|
| Fluency floor | RCK v1 README; 10^25 FLOPs argument | v0 toy scale only. Capability surfaces present, not consumer-grade. |
| "Illusion of state" expressiveness ceiling | arXiv:2404.08819 | Multi-signal EFE decoder draws from 7 sources, not just the recurrent state. Sidestep via ensembling. |
| Associative recall deficit | RWKV/Mamba weakness, HYMN inherits | Sharded HRR KB does explicit associative recall as one of 7 EFE signals. |
| Long-context >32K | Pure non-transformer weakness | v0 ≤8K; long-context is v2 via federated memory bundling. |
| PCN depth instability beyond ~100M | Pinchetti 2025, arXiv:2510.23323 | PCN is one component, not the whole stack. HYMN/FEP/Tsetlin carry load when PCN saturates. |

---

## 2. Component selection

70% of components are existing, validated code from five prior NORTHTEKDevs repos. 30% is integration code plus capability additions. Full component map lives in `docs/architecture/component-map.md`.

### From RCK (~/projects/active/rck/) — the substrate

`relational.py`, `knowledge_base.py`, `pcn.py`, `liquid_state.py`, `tsetlin.py`, `fep.py`, `bigram.py`, `efe.py`, `generative.py`, `tokenizer.py`, `self_model.py`, `theory_of_mind.py`, `introspect.py`, `metacog.py`, `dialogue.py`, `inference.py`, `compose.py`, `explain.py`, `compose_answer.py`, `think_aloud.py`, `session.py`, `conscious_agent.py`, `bulk_ingest.py`, `synonyms.py`. **Cut from RCK:** `server.py`, `mcp_server.py` (move to separate `rain-serve` crate later).

### From Hyperion (~/projects/active/hyperion/) — the sequence engine

`vsa_core/` (10K-dim bipolar primitives, replaces RCK's 4K FHRR), `hymn/` (MLP update rule replacing attention), `pure_vsa/` (ablation baselines), `train_hymn_mini.py`, `train_hymn_scan.py`, `eval_scan.py`. **Cut:** `train_nanogpt_baseline.py` (lives in evals/baselines/), `sdk-ts/`, Docker stuff.

### From SOFAR (~/projects/active/sofar/) — routing transplanted onto VSA

`mapper.py` (SVD over codebook + role matrices instead of W_q/W_k/W_v/W_o), `encoder.py` (FrequencyLayeredEncoder LOW/MID/HIGH bands), `attention.py` (LoRA + beam-steering envelopes), `benchmarks/fidelity.py`. **Cut:** HuggingFace-transformers-specific code (`_w_read`/`_w_write` Conv1D handling, hook plumbing, GPT-2/Llama patch examples).

### From Cognitive Kernel Polyglot — operational substrate

`core-bridge.ts` + WASM/Rust/Go cores, `evolution.{ts,rs,go}` (NSGA-II Pareto), `calibration.ts` (Bayesian shrinkage), `crystals.ts` + schema, `workload-observer.ts`, `failed-learning-queue.ts`, `bshr-loop.ts`, `decision-router.ts`, `memory.ts`, `response-cache.ts`, embedding-recall (G6). **Cut:** `llm-gateway.ts`, `local-model.ts` (any module that calls an external LLM — RAIN has none).

### From Evolve (~/projects/active/evolve/) — promotion gate

`evolve-core/src/agent_config.rs`, `schema.rs`, `ids.rs`, champion/challenger Bayesian beta-binomial promotion. SQLite at `~/.rain/db.sqlite`. **Cut:** planned axum proxy + dashboard crates.

### Hard out — not in v0

- **aiproof** — separate input-side linter, optional, not load-bearing.
- **Rhizome** — federated VSA transport, deferred to v2.

### Built new

1. `rain/bridge.py` — FEP-unified objective combining every component's local-loss signal.
2. `rain/spine.py` — HYMN-FEP fused sequence engine.
3. `rain/heads/` — capability heads (dialogue, code, tool-use, reasoning, generation).
4. `rain-rs/` — unified Rust core (HYMN forward, VSA bind/bundle, Tsetlin batch, SOFAR SVD, NSGA-II).

### Locked architectural decisions

- **Binding convention:** Kanerva bipolar 10K-dim (Hyperion's primitives). RCK code adapts up from 4K-dim FHRR.
- **Calibration source-of-truth:** polyglot kernel `calibration.ts` (tenant-scoped, has Bayesian shrinkage).
- **Tokenizer:** SentencePiece BPE 32K vocab (v0).
- **Substrate language:** Python 3.11+ with Rust hot kernels via PyO3.

---

## 3. Data flow / inference pipeline

One user-prompt → one output-token cycle, numbered steps mapping 1:1 to modules.

**Step 0 — Tokenize + encode.** Text → tokenizer → token IDs → codebook hypervectors → bound into position-aware bundle. State `s_0` is bipolar 10K-dim.

**Step 1 — Frequency-band decomposition.** `sofar/encoder.py` partitions `s_0` into LOW/MID/HIGH via Hadamard (not FFT — bipolar HVs aren't smooth). Matches FEP hierarchical generative model layers.

**Step 2 — SVD routing axes.** `sofar/mapper.py` runs SVD over codebook + role matrices (cached, recomputed on KB updates). Extracts top-k principal routing directions.

**Step 3 — Beam-steering envelope.** `sofar/attention.py` selects FOCUS/SWEEP/TRACK based on dialogue context + introspection state. Energy-normalized envelope applied along chosen SVD directions.

**Step 4 — HYMN MLP state update.** `hymn/` produces `s_1` via bipolar MLP update rule. The transformer-attention replacement.

**Step 5 — Seven parallel candidate sources query `s_1`:**

| # | Source | Module | Produces |
|---|---|---|---|
| 1 | HYMN prediction | `hymn/` | Codebook nearest-neighbor of `s_1`. |
| 2 | LSM recall | `core/liquid_state.py` | RLS-based recurrent prediction. |
| 3 | Bigram VSA | `core/bigram.py` | N-gram suggestion. |
| 4 | FEP-acted | `core/fep.py` | Action minimizing Expected Free Energy. |
| 5 | KB lookup | `core/knowledge_base.py` | Sharded HRR fact recall. |
| 6 | Tsetlin vote | `core/tsetlin.py` | Per-clause syntactic-validity vote. |
| 7 | Crystal recall | polyglot `crystals.ts` | Cosine-nearest past crystal continuation. |

**Step 6 — EFE decoder fuses, calibration re-weights, anti-repetition penalizes.** `core/efe.py` fuses with workload-observer-tuned weights (NSGA-II evolved). `cognition/metacog.py` applies per-relation confidence floor/amplification. Anti-repetition penalties (single-char, bigram-cycle, trigram-cycle). Top-p sampling.

**Step 7 — Output assembly.** Confidence + epistemic class (know/think/guess/unknown). `unknown` → refusal/hedge + learning mode. `know/think` → sampled token. Reasoning mode emits CoT trace via `think_aloud.py` + citations via `explain.py`. Tool-use mode: FEP candidate is an action, polyglot kernel dispatches it.

**Step 8 — Append + loop.** Token appended, codebook bind into `s_2`, return to Step 1. Until EOS / max length / epistemic exhaustion.

**Step 9 — Inline local learning (no backprop).** Per token: PCN Hebbian on codebook, LSM RLS update, Tsetlin clause feedback, FEP rank-1 A update. Per turn: KB writes for new facts, crystal write-through. Total cost ~5-10ms CPU per token (RCK v1.0 baseline).

**Step 10 — Population evolution (background, non-blocking).** Once per N turns: NSGA-II generates candidate HYMN/SOFAR/Tsetlin variants. Shadow inferences on probes. Champion/challenger Bayesian beta-binomial promotion.

### Properties

- Latency: 5-10ms/token CPU at toy scale (RCK v1.0 baseline); 1-2ms with Rust hot path.
- Memory: O(K_shards × D) + O(D × |HYMN MLP|). ~5GB resident at 10K state + 64 shards × 2000 facts.
- Continual learning: every token writes; nothing frozen at runtime.
- Transparency: every output has confidence + epistemic class + (reasoning mode) citation chain.
- All 5 capability surfaces (gen / dialogue / tool / code / reasoning) share this pipeline with different head configs.

### Gap-closers added to keep LLM-equivalent UX

1. **Codebook warm-start from public pretrained embeddings** (FastText/sentence-transformers projected to bipolar). Not an LLM-substrate — just smart init.
2. **Bootstrap-only backprop on HYMN spine.** Initial corpus training phase only; weights frozen after, switch to local rules.
3. **BPE 32K tokenizer at v0** (not v0.5).
4. **`rain-chat` LLM-equivalent chat UI.** Tauri + React + shadcn. Looks identical to ChatGPT/Claude. RAIN's advantages surface as features (confidence chips, citation toggle, "RAIN learned X this turn" banners).

### Ten capability additions

| # | Addition | What it adds | From |
|---|---|---|---|
| 5 | KB-RAG-as-decode-signal | Open-domain Q&A competitiveness via first-class EFE signal | RCK + polyglot |
| 6 | Native "thinking" mode | o1/Claude-extended-thinking UX | RCK think_aloud + introspect |
| 7 | Speculative decoding | 2-4x perceived speedup | RCK existing modules |
| 8 | Persistent agent-identity layer | Per-user style learning — structural moat | Evolve |
| 9 | Federated VSA bundle as built-in primitive | Mix B path baked in | RCK v1.1 federated bundling |
| 10 | Sleep / replay consolidation | Hippocampal-replay analog | New module, existing primitives |
| 11 | Self-verification head | Architectural Constitutional-AI | RCK explain.py |
| 12 | Tool registry + function calling | Agentic capability surface | Evolve + polyglot + RCK FEP |
| 13 | Values / constitution layer | Structural alt to RLHF | RCK ToM |
| 14 | Multimodal hooks (v0 interface, v2 impl) | VSA is modality-agnostic | RCK + new encoders |

---

## 4. Training protocol + FEP-unified objective

### The unifying objective

`F = -E_q[log p(o,s)] - H[q(s)] = KL(q(s)||p(s)) - E_q[log p(o|s)]`. Every RAIN component is a piece of computing `F` or sampling from `q`. Full derivation in `docs/architecture/fep-unified-objective.md`.

### Component → FEP role

| Component | FEP role |
|---|---|
| HYMN MLP | Amortized approximate posterior `q(s_{t+1} | s_t, o_t)`. |
| SOFAR SVD + beam-steering | Precision weighting on residual dimensions. |
| Frequency-band encoder | Hierarchical generative model layers `p(s^L) p(s^M|s^L) p(s^H|s^M)`. |
| Sharded HRR KB | Long-term grounded prior `p(s)`. |
| Crystals + embedding recall | Episodic prior `p(s|context)`. |
| PCN | Local layer-wise prediction-error minimization. |
| LSM with RLS | Recurrent posterior regression. |
| Tsetlin clauses | Symbolic prior contribution `log p(s)`. |
| FEP rank-1 A | Online generative-model parameter update. |
| ToM beliefs | Conditional prior `p(s|other_agent_model)`. |
| Self-model | Conditional prior `p(s|self_state)`. |
| Metacog calibration | Posterior uncertainty `H[q(s)]` modulating sampling. |
| EFE decoder | Action selection minimizing G(π). |
| NSGA-II evolution | Posterior model selection across structures. |
| Champion/challenger | Bayesian model averaging with promotion threshold. |
| Sleep / replay | Offline F minimization on stored crystals. |
| Self-verification | Posterior consistency check. |

### Phase 1 — Bootstrap (expensive, one-shot, blank-slate → fluency floor)

| Step | What | Time | Cost |
|---|---|---|---|
| 1.1 | Codebook warm-start from FastText/sentence-transformers via sign projection | minutes | $0 |
| 1.2 | KB seeding from Wikidata + ConceptNet + commonsense (50K facts) | hours | ~$5 |
| 1.3 | HYMN gradient pre-train on corpus (next-token NLL). **Only place backprop appears.** Weights frozen after. | 1-2w v0 / 1mo v0.5 / 2mo v1.0 | $200-500 / $2k / $20-50k |
| 1.4 | SOFAR adapter identity-at-init | seconds | $0 |
| 1.5 | Tsetlin clause seeding (Type-I training on same corpus) | days | $0 (CPU) |
| 1.6 | LSM RLS init (analytical single-pass) | hours | $0 (CPU) |
| 1.7 | FEP A matrix init from PCA of residual stream during 1.3 | 0 | $0 |

### Phase 2 — Continual (cheap, forever, no backprop)

After Phase 1 freezes HYMN, **no global gradient ever fires again**. All learning local:

- Per-token: PCN Hebbian on codebook, FEP rank-1 on A, Tsetlin clause feedback, LSM RLS, crystal write-through.
- Per-turn: KB writes for new facts, ToM updates, self-model updates, calibration tally.
- Per-night: crystal replay via PCN consolidates episodic → structural.
- Per-week (background): NSGA-II + champion/challenger over shadow probes.

Phase 2 cost is essentially zero per user.

### Corpus and hardware per phase

| Phase | Corpus | KB | Hardware | Wall time | Cost |
|---|---|---|---|---|---|
| v0 | Tiny Shakespeare + open dialogue subset + TinyCodes (~60MB) | 12K facts (2K RCK + 10K Wikidata) | Single 4090 + CPU | 1-2 weeks | $200-500 |
| v0.5 | WikiText-2 + Stack subset + curated dialogue | 50K facts | Single 4090/A100 | ~1 month | ~$2000 |
| v1.0 | WikiText-103 + Pile subset + code corpus | 1M facts | 8x A100/H100 | 1-2 months | $20-50k |
| v2.0 | per-specialist | per-specialist | N parallel single-GPU | ~1w/specialist | ~$5-10k/specialist |

### Honest risks + mitigations

1. **HYMN gradient pre-train doesn't reach baseline NLL.** Risk medium. Mitigation: increase rank, curriculum, or fall back to standard transformer for spine *Phase 1 only* with Phase-1-hot-swappable architecture.
2. **Continual drift after weeks.** Risk medium. Mitigation: sleep/replay, KB hygiene, calibration drift monitoring.
3. **SOFAR-on-VSA transplant doesn't help.** Risk medium. Mitigation: SOFAR is Phase-1 ablation; drop if no val-loss improvement.
4. **NSGA-II finds local optima.** Risk low. Mitigation: champion/challenger guard; manual override.
5. **Inline learning slows inference.** Risk low. Mitigation: configurable; background mode available.

---

## 5. v0 scope + acceptance criteria

Working `rain` Python package + Rust core + chat UI on single RTX 4090 + CPU, passing 18 falsifiable benchmarks across 3 tiers.

### Tier 1 — Five novelty-surface benchmarks

| # | Surface | Benchmark | Threshold | Frontier LLM | RCK status |
|---|---|---|---|---|---|
| N1 | Continual learning | A→B→A retention on disjoint 5K-fact tasks | ≥0.50 | <0.30 (CSUR 2025) | 0.50 measured v1.0 |
| N2 | Calibrated uncertainty | ECE on 14 relation types, 1000Q probe | ≤0.05 | 0.15-0.25 (Lin 2022) | ≤0.05 measured v1.3 |
| N3 | Compositional gen | SCAN add-prim + 3/4/5-slot combos | 100% / ≥95% / ≥90% / ≥85% | GPT-4 ~70% on add-prim | 100% measured v1.1 |
| N4 | Structural ToM | Sally-Anne + 20-turn BDI tracking | 10/10 + 18/20 | LLMs fail ~40% (Ullman 2023) | Module exists |
| N5 | Grounded transparency | 100-claim audit, citation coverage | ≥98%, ≤1% hallucination | LLMs ~30-50% with RAG | explain.py exists |

### Tier 2 — Eight LLM-parity-at-toy-scale benchmarks

| # | Benchmark | Threshold | Baseline |
|---|---|---|---|
| L1 | Tiny Shakespeare val-loss | ≤1.55 nats/char | nanoGPT-char matched FLOPs |
| L2 | WikiText-2 perplexity (small subset) | ≤80 | nanoGPT-medium ~80 |
| L3 | HumanEval-tiny (10 problems) | ≥40% pass@1 | GPT-2 ~10% |
| L4 | GSM8K-tiny (5 problems) | ≥40% | Pythia-160M ~15% |
| L5 | MMLU-tiny (150 questions) | ≥30% | Pythia-410M ~28% |
| L6 | MTBench-tiny (10 prompts, judged) | ≥3.0/10 | nanoGPT-medium ~2.0 |
| L7 | Tool-use (10 function-call tasks) | ≥8/10 | GPT-3.5 ~9/10 |
| L8 | Inference latency CPU | ≤10ms/token | RCK v1.0 measured 5.5ms |

### Tier 3 — Five architectural-soundness checks

| # | Check | Threshold |
|---|---|---|
| A1 | EFE source-mix sanity | Each of 7 candidate sources contributes ≥5% on balanced probe |
| A2 | SOFAR-on-VSA improvement | Val-loss with routing < without, by ≥3% |
| A3 | NSGA-II promotion rate | ≥1 champion-displacing variant per 100 generations over 1000 gens |
| A4 | Local-rule stability | No NaN, ECE drift ≤0.1, no KB corruption over 10K turns |
| A5 | Polyglot dispatch correctness | WASM/Rust/Go/TS bit-identical (modulo float ε) |

### Kill criteria

| Trigger | Response |
|---|---|
| 8 weeks Phase 1, HYMN val-loss > 1.1× nanoGPT | Swap HYMN for standard small transformer Phase 1 only. Phase 2 unchanged. |
| A2 fails | Drop SOFAR routing, use uniform in EFE. |
| A3 fails | Drop NSGA-II, manual hyperparam tuning. |
| A4 fails | Tighten sleep/replay, gate continual behind manual trigger if 3 attempts fail. |
| Tier-1 surface lost to a frontier LLM | Pivot to the *bundle* claim rather than individual claims. |
| Tier 2 L1 fails | Architecture broken — stop and rederive. Canary. |

### Comparative bar against non-LLM alternatives

Match or beat RWKV-7-mini + LFM2-small + Mamba-3-small + Pythia-160M on Tier 2 at matched FLOPs while dominating them on Tier 1 (which they don't have).

### v0 timeline

| Week | Milestone |
|---|---|
| 1 | Repo scaffold, polyglot + RCK + Hyperion integrated, Phase 1.1-1.2 running |
| 2-3 | Phase 1.3 HYMN pre-train on Tiny Shakespeare, hit L1 |
| 4 | Phase 1.4-1.7 finish, full pipeline working |
| 5 | Tier 3 (A1-A5) green |
| 6-7 | Tier 1 (N1-N5) green |
| 8 | Tier 2 (L1-L8) green |
| 9-10 | 10 capability additions integrated |
| 11-12 | Polish, chat UI, docs, **private v0 tag** |

### v0 artifacts

- `rain` Python package (private pip)
- `rain-rs` Rust core (PyO3 + WASM + native FFI)
- `rain-chat` Tauri desktop chat UI
- `rain-cli` CLI for batch eval + REPL
- Private GitHub repo on NORTHTEKDevs
- Design doc + funding waypoints + 18-benchmark eval suite + reproducibility notebook
- 5-minute demo video: continual learning, calibrated answer, compositional task, ToM story, grounded explanation

---

## 6. Repo structure + tech stack

### Identity

- Name: `rain` (working codename)
- Org: NORTHTEKDevs, PRIVATE
- License: Proprietary, SPDX NOASSERTION
- Headers: `CONFIDENTIAL\n(c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io` on every source file
- Branch protection, squash-only, Dependabot, secret scanning (matches SOFAR config)

### Tech stack

| Layer | Tech | Rationale |
|---|---|---|
| Core | Python 3.11+ | RCK + Hyperion + SOFAR lineage; fastest iteration |
| Hot kernels | Rust 2024 (MSRV 1.85) + PyO3 | Matches Evolve + polyglot; perf for VSA/HYMN/Tsetlin/SOFAR/NSGA-II |
| Browser/Edge | WASM via wasm-bindgen | Same Rust core compiles to WASM; matches polyglot kernel |
| Orchestration | TypeScript (Node + browser) | Reuses cognitive-kernel-polyglot directly |
| Chat UI | Tauri 2 + React + shadcn | Matches Rhizome's Tauri patterns; cross-platform |
| Tokenizer | SentencePiece BPE 32K | Standard; closes rare-word gap |
| Storage | SQLite via sqlx (local), Postgres via polyglot kernel adapter (cloud) | Matches Evolve + polyglot |
| CI | GitHub Actions matrix (Linux/macOS/Windows × Py 3.11/3.12/3.13) | Matches SOFAR + polyglot |
| Build | maturin (Py+Rs), wasm-pack (browser), cargo (Rs), pnpm (TS) | Already in use |

### Initial scaffold steps (executed as part of this design approval)

1. Create directory tree under `~/projects/active/rain/`
2. Vendor source map in `VENDORED.md` (no eager copying)
3. Write design docs, README, LICENSE, SECURITY, CHANGELOG, CONTRIBUTING, CRYSTAL
4. Generate copyright headers on stub files
5. `git init`, first commit "scaffold + design docs"
6. `gh repo create NORTHTEKDevs/rain --private`
7. Push initial commit
8. Apply repo lockdown matching SOFAR
9. CI workflow runs initial empty-suite to confirm scaffold sound
10. Tag `v0.0.0-scaffold`
11. Invoke `superpowers:writing-plans` for the 12-week implementation plan

---

## Appendices

- **A — FEP unified objective math:** `docs/architecture/fep-unified-objective.md`
- **B — Component map:** `docs/architecture/component-map.md`
- **C — Benchmark suite spec:** `docs/architecture/benchmark-suite.md`
- **D — SOFAR-on-VSA transplant derivation:** `docs/design/sofar-on-vsa-transplant.md`
- **E — Multi-signal EFE decoder:** `docs/design/multi-signal-efe-decoder.md`
- **F — Funding waypoints:** `docs/plans/2026-05-22-rain-funding-waypoints.md`
- **G — Implementation plan (writing-plans output):** `docs/plans/2026-05-22-rain-implementation-plan.md`

---

## Provenance

- Source repos vendored: rck (~/projects/active/rck/), hyperion (~/projects/active/hyperion/), sofar (~/projects/active/sofar/), cognitive-kernel-polyglot (~/projects/active/cognitive-kernel-polyglot/), evolve (~/projects/active/evolve/)
- Prior-art research pass: 2026-05-21 (28 papers, 8 question dimensions)
- Brainstorm session: 2026-05-22 via superpowers:brainstorming
- Hard out: aiproof, Rhizome (deferred to v2 + optional)
