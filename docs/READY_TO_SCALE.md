# RAIN v0 -> v1.0 -- The Scaling Bridge

> What separates the working v0 demo (L1 NLL = 1.47, chat REPL with KB +
> HYMN + RLAIF) from a real production model. Honest, itemized, and
> aimed at "press button" execution -- so we know exactly what to do
> when compute lands.

## Where we are (v0, shipped this shift)

| Capability | Status | Notes |
|---|---|---|
| HYMN char-level LM | DONE | L1 NLL 1.47 on Tiny Shakespeare, passes 1.55 target |
| HYMN trained on WikiText-2 | DONE | self-eval NLL 1.72 (carry=16 30K, 11 min DirectML) |
| HYMN trained on WikiText-103 | DONE | self-eval NLL 2.04 (carry=8 100K, 20 min DirectML) -- 4979-char vocab, uniform baseline 8.51, so 24% of uniform |
| Sequence-carry training | DONE | the lever that unlocked L1 |
| AdamW + warmup + cosine + grad_clip | DONE | tuned defaults in runbook |
| NLL loss via codebook softmax | DONE | replaces v0 MSE-on-HV reference |
| KB distillation from local Ollama | DONE | 2711 facts, then judge-filtered to 2399 |
| LLM judge feedback loop (RLAIF) | DONE | 50/50 verdicts in scaled run |
| ConsciousAgent with HYMN fallback | DONE | new in this shift -- attach_hymn_sampler() |
| Chat REPL combining all the above | DONE | scripts/rain_chat |
| Bootstrap orchestrator (Phase 1.1-1.7) | DONE | rain.train.bootstrap end-to-end |
| Schema v2 self-describing checkpoints | DONE | L1 auto-picks eval mode |
| 200-row test suite | DONE | 203 passing |
| Runbook + quickstart + capabilities docs | DONE | docs/training-runbook.md, docs/quickstart.md, docs/CAPABILITIES.md |

## What v1.0 needs (the scaling bridge)

In rough order. Each item is gated -- you can't skip ahead.

### 1. Real warm-start (the missing Phase-1 step) -- DONE for chars, partial for BPE

The bootstrap orchestrator's `warm_start_from_vectors` accepts a dict of
{token -> embedding} but the v0 runs all skipped it and started from
random codebook vectors. **Now wired both ways:**

- `rain.train.warm_start_chars` -- feature-based char prior. Zero deps.
  3K-step A/B on Tiny Shakespeare: final_loss=2.40 vs random 2.87 (-16%).
  Cluster gap (within-letter cos minus letters-to-digits cos) = 0.74.
- `rain.train.warm_start_minilm` -- sentence-transformers MiniLM (~80 MB).
  **Negative finding for chars:** final_loss=2.73 vs feature 2.40.
  MiniLM's single-char embeddings collapse on sentence-pooling so
  the cluster gap is only 0.01. **Keep MiniLM for BPE subwords** via
  `warm_start_bpe_minilm()` -- that's where each token is genuinely
  a "sentence" and MiniLM shines.

Bootstrap entry: `bootstrap_phase1(warm_start_method="chars"|"minilm")`.
CLI flag on `pretrain_hymn_torch`: `--warm-start-chars` or
`--warm-start-minilm`.

**Status:** wired, tested, A/B-validated. Char path is the v0 winner.

### 2. Scale corpus to WikiText-103 (or larger) -- DONE for download

`data/corpora/wikitext103_train.txt` is on disk @ 543 MB (~45x WT-2).
Extraction script: `scripts/extract_wikitext103.py`.

**Compute remaining:** 8-16 hours local training on the workstation for
one pass. This is the next button-press.

### 3. Real Phase-1 gradient pre-train end-to-end

Wire the existing `scripts.pretrain_hymn_torch` to consume the bootstrap
orchestrator's output (codebook + tokenizer) instead of building its own.
Single CLI command that runs:
- bootstrap_phase1(corpus, KB, warm_start_vectors)
- pretrain HYMN for N steps with the carry=16 + NLL recipe
- freeze the checkpoint
- emit a Phase-1-complete bundle

**Estimated effort:** half a day. Compute: 8-48 hours local for
WikiText-103-scale; 0-$50 cloud GPU if we want it faster.

### 4. dim=2048 retune (validate the LR-scaling negative finding)

dim=2048 at the dim=1024-tuned hparams got L1=1.88 vs dim=1024's 1.47.
Currently running dim=2048 with lr=2e-4 (smaller LR per scaling law) --
expected to outperform the 1.88. After this lands we know the right
LR-vs-dim relationship.

**Estimated effort:** running now; finishes in ~30 min. Lands the
empirical scaling rule.

### 5. Integrate cognitive primitives during inference -- DONE except SOFAR LoRA

The LSM / Tsetlin / FEP / SOFAR modules are TRAINED and TESTED in
isolation (Tier-3 A1-A5 all wired). Inference-path integration:

- LSM: per-turn state update from token embeddings -> reservoir state
  -> RLS readout. **DONE.** Write side in `agent.tell`; read side
  exposes `lsm_state_l2` on every Answer when `enable_continual=True`.
- Tsetlin: per-relation clause votes -> additional epistemic signal.
  **DONE.** Type-I feedback on every tell; `tsetlin_max_abs_vote`
  exposed on every Answer.
- FEP rank-1 A matrix: low-rank update of the generative model after
  each turn. **DONE.** `fep_cos` exposed on every Answer.
- Blended `cognitive_agreement` score (0..1) derived from the three
  signals. Exposed on Answer.cognitive_signals for downstream gating.
  **DONE.**
- SOFAR beam adapter: trained LoRA weights focus state on relevant
  routing directions. This is what unlocks the A2 ablation kill-trigger
  (currently 0.08% improvement; design target >=3%). **NOT STARTED.**

**Status:** 4/5 signals live with 6 passing tests in
`tests/test_agent_cognitive_signals.py`. Only SOFAR LoRA training
remains (several days of training + ablation re-runs).

### 6. Train on conversational data so HYMN can actually chat

HYMN trained on Shakespeare or WikiText answers KB-miss queries with
period-correct prose, not factual answers. For RAIN to be a useful
chat agent, HYMN needs to be pre-trained on Q/A or dialogue data:

- Alpaca (50K Q/A pairs, ~40 MB).
- ShareGPT (filtered to ~10K curated dialogues).
- Or use the local Ollama Qwen-30B to generate a synthetic Q/A dataset
  from RAIN's own KB.

**Estimated effort:** 1-2 days to wire + train. Compute: similar to L1
training time.

### 7. Larger KB (~50K-100K facts)

Current: 2399 judge-filtered facts across 120 topics. v1.0 target:
50K-100K facts spanning Wikipedia category breadth.

- Run scripts.seed_kb_from_ollama on a 1000-topic list.
- Run scripts.filter_seed_with_judge with a better judge if available
  (the llama3.2:3b judge is sometimes wrong itself; qwen-30B would do
  better if its empty-response bug can be worked around).
- Estimated 24-48 hours wall on local Ollama.

**Estimated effort:** 0 dev time; pure compute. ~48 hours wall.

### 8. Deployment surface

The chat REPL is the v0 UI. v1.0 needs:
- HTTP/WS server wrapping ConsciousAgent.ask + sampler. **DONE.**
  `scripts/rain_server.py` -- aiohttp server with /health, /ask, /tell,
  /describe, /sample, /judge, /tally, /snapshot, /self endpoints.
  +7 tests in `tests/test_rain_server.py`. CLI flags compose every
  working surface (--enable-continual, --use-rag, --judge-model).
- Optional Tauri 2 + React shell per the design plan. **NOT STARTED.**
- Multi-user session management. **NOT STARTED.**

**Estimated effort remaining:** 1 week for Tauri UI, several days for
multi-user state isolation. Backend is done.

## "Press the v1.0 button" cost estimate

Best-case (all local, no cloud GPU):
- (1) warm-start: $0
- (2) WikiText-103 download: $0
- (3) Phase-1 pretrain: $0 (8-48 hrs local CPU + iGPU)
- (4) dim=2048 retune: $0 (1-2 hrs CPU)
- (5) cognitive integration: dev time only, $0
- (6) conversational HYMN: $0 (Alpaca + local training)
- (7) KB scale-up: $0 (Ollama runs 24-48 hrs)
- (8) deployment: $0 dev (hosting separate)

**Total compute: $0. Total dev time: ~3-4 weeks.**

Cloud-accelerated (if speed matters):
- WikiText-103 pretrain on a 4090 rental: $20-50.
- WikiText-103 pretrain on A100: $50-150.
- Anything bigger (C4 subset, code datasets, etc.): $200-1000.

This is the OPPOSITE of the funding-plan estimate of $20-50k. The
design wins via local-compute + structural priors + KB grounding,
not via cluster scale.

## What v1.0 will be able to claim

After items 1-6 land:

- L1 NLL well below 1.0 on Tiny Shakespeare (with sequence-carry +
  warm-start + bigger HYMN -- the gap from 1.47 to ~0.9 is mostly
  warm-start).
- L2 NLL competitive with char-RNN baselines on WikiText-2.
- Conversational HYMN that answers free-text questions, with KB-grounded
  citations when available and HYMN guesses tagged as such.
- Working RLAIF loop that improves calibration over time without human
  labels.
- Demonstrable continual learning (N1), calibration (N2), compositional
  generalization (N3), ToM (N4), grounded transparency (N5).
- Total compute spend: under $200.

## What v1.0 will NOT be

- Not GPT-4-class fluency. The model is small.
- Not a single-model competitor to Claude/Gemini -- different design
  thesis (KB-grounded + cognitive surfaces > pure-LM).
- Not multimodal (vision/audio queued for v2+).

## TL;DR

We're at the boundary. v0 demo works. The next 3-4 weeks of work is
itemized in this doc and almost entirely zero-marginal-cost on the
existing workstation. The v1.0 "press button" doesn't need any new
research, just sequenced engineering.

Item ordering for the next shift:
1. dim=2048 lr=2e-4 result (running now -- ~30 min)
2. Wire FastText warm-start into bootstrap_phase1
3. Download WikiText-103
4. Single end-to-end Phase-1 pretrain run
5. Then everything else.
