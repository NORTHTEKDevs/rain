# RAIN Component Map

**Companion to:** `docs/plans/2026-05-22-rain-design.md` (§ 2), `VENDORED.md`
**Status:** Authoritative crosswalk between RAIN modules and their source repos.

For full source-to-destination file mappings + adapter notes, see
`VENDORED.md`. This file is the higher-level conceptual map.

---

## Layered view of RAIN

```
+----------------------------------------------------------+
| rain-chat (Tauri + React + shadcn)                       | <- LLM-equivalent UX
+----------------------------------------------------------+
| rain-cli (Python)                                        | <- REPL + batch eval
+----------------------------------------------------------+
| rain-kernel (TS, re-exports cognitive-kernel-polyglot)   | <- Browser + server orchestration
+----------------------------------------------------------+
| rain (Python package)                                    |
|   model.py            <- top-level                       |
|   agent.py            <- ConsciousAgent (RCK)            |
|   spine.py            <- HYMN-FEP sequence engine (NEW)  |
|   bridge.py           <- FEP unified objective (NEW)     |
|   heads/              <- capability surfaces (NEW)       |
|   additions/          <- 10 capability adds              |
|   routing/            <- SOFAR transplanted              |
|   cognition/          <- RCK self_model / ToM / metacog  |
|   core/               <- RCK substrate                   |
|   train/              <- Phase 1 + Phase 2 + Evolve      |
|   tokenize/           <- BPE 32K                         |
|   data/               <- bulk_ingest + synonyms          |
|   session.py          <- persistence via crystals        |
|   config.py           <- workload profiles               |
+----------------------------------------------------------+
| rain-rs (Rust, PyO3 + WASM)                              |
|   vsa.rs / hymn.rs / tsetlin.rs / sofar.rs / nsga2.rs    | <- hot kernels
+----------------------------------------------------------+
```

---

## Five capability surfaces — same pipeline, different head configuration

| Surface | Head module | What it specializes |
|---|---|---|
| Generation | `rain/heads/generation.py` | Token emission with anti-repetition + top-p sampling defaults. |
| Dialogue | `rain/heads/dialogue.py` | Multi-turn DialogueContext + ToM updates per turn. |
| Code | `rain/heads/code.py` | Code-tokenizer + structural Tsetlin clauses for syntax validity. |
| Reasoning | `rain/heads/reasoning.py` | `think_aloud` CoT trace + `explain` citation chain + BSHR loop activation. |
| Tool use | `rain/heads/tool_use.py` | FEP candidate is an action; polyglot kernel dispatches the tool call; result is the next observation. |

All five share `rain/spine.py` (the HYMN-FEP sequence engine) and
`rain/core/efe.py` (the 7-source multi-signal decoder). Differences are only
in head configuration.

---

## Ten capability additions — placement map

| # | Addition | Location |
|---|---|---|
| 5 | KB-RAG-as-decode-signal | `rain/additions/kb_rag.py` |
| 6 | Native thinking mode | `rain/additions/thinking_mode.py` |
| 7 | Speculative decoding | `rain/additions/speculative.py` |
| 8 | Persistent agent identity | `rain/additions/identity.py` (Evolve AgentConfig wrapper) |
| 9 | Federated VSA bundle primitive | `rain/additions/federated.py` |
| 10 | Sleep / replay consolidation | `rain/additions/sleep_replay.py` |
| 11 | Self-verification head | `rain/additions/self_verify.py` |
| 12 | Tool registry + function calling | `rain/additions/tool_registry.py` |
| 13 | Values / constitution layer | `rain/additions/values.py` |
| 14 | Multimodal hooks (interface only at v0) | `rain/additions/multimodal_stub.py` |

---

## Seven EFE-decoder candidate sources — origin map

| # | Source | Module | Origin |
|---|---|---|---|
| 1 | HYMN prediction | `rain-rs/src/hymn.rs` (forward) | Hyperion `hymn/` |
| 2 | LSM recall | `rain/core/liquid_state.py` | RCK `rck/liquid_state.py` |
| 3 | Bigram VSA | `rain/core/bigram.py` | RCK `rck/bigram.py` |
| 4 | FEP action | `rain/core/fep.py` | RCK `rck/fep.py` |
| 5 | KB lookup | `rain/core/knowledge_base.py` | RCK `rck/knowledge_base.py` |
| 6 | Tsetlin vote | `rain/core/tsetlin.py` (+ Rust hot path) | RCK `rck/tsetlin.py` + new `rain-rs/src/tsetlin.rs` |
| 7 | Crystal recall | via `rain-kernel` re-export of polyglot `crystals.ts` | polyglot kernel |

All seven get fused in `rain/core/efe.py` (RCK's existing multi-signal
decoder, extended from 4 sources to 7).

---

## SOFAR transplant placement

| SOFAR original | RAIN target | Transformation |
|---|---|---|
| `sofar/mapper.py` (SVD of W_q/W_k/W_v/W_o) | `rain/routing/mapper.py` | Targets codebook + role matrices instead of transformer weight matrices. Math identical (SVD of read+write); domain different. |
| `sofar/encoder.py` (FrequencyLayeredEncoder, LOW/MID/HIGH) | `rain/routing/encoder.py` | Hadamard partition of bipolar 10K-dim state replaces FFT-based spectral partition. |
| `sofar/attention.py` (LoRA + beam_gate + FOCUS/SWEEP/TRACK envelopes) | `rain/routing/beam.py` | LoRA adapter rank-projected onto routing directions; identity-at-init preserved; envelopes carry over unchanged. |
| `sofar/benchmarks/fidelity.py` | `evals/tier3_soundness/sofar_fidelity.py` | Re-targeted at "did routing preserve VSA-state symbolic content?" |

---

## What's NOT mapped (out of v0)

- **aiproof** — never imported. Optional input-side linter applied externally
  to user prompts; can be wired up later via `rain-chat` UI hook.
- **Rhizome** — never imported. Federated VSA bundle PRIMITIVE is exposed
  (via `rain/additions/federated.py` from RCK v1.1), but the transport
  layer is deferred to v2.

---

## Verification checklist before declaring a module "integrated"

For each vendored module:

- [ ] Source file copied to `third_party/<repo>/<path>` with provenance hash
      logged in `VENDORED.md`.
- [ ] Adapted copy placed at the RAIN destination per this map.
- [ ] Header on adapted copy carries the RAIN confidential notice.
- [ ] Original-source attribution preserved in a `# Original-source:` comment.
- [ ] Bind-space adapted (4K complex FHRR → 10K bipolar) if from RCK.
- [ ] All transformer-specific imports stripped if from SOFAR.
- [ ] All LLM-gateway imports stripped if from polyglot kernel.
- [ ] Integration test added covering the module's role in the RAIN pipeline.
- [ ] PR references the section of the design doc + this map + VENDORED.md
      entry.
