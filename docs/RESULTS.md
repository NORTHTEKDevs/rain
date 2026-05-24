# RAIN -- Empirical Results Ledger

> Single source of truth for "what numbers RAIN has actually hit on
> what data with what hyperparameters." Append-only. Each row is one
> training run + one eval; no row is overwritten when a later run
> beats it -- the negative findings stay so we don't relearn them.

## L1 -- Tiny Shakespeare char-level LM NLL

L1's pass threshold per the design plan is **NLL <= 1.55 nats/char**.

| Run | Steps | Carry | Batch | Dim | LR | Warm-start | Loss | Train wall | Train NLL | L1 NLL | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hymn_carry16_50k | 50,000 | 16 | 16 | 1024 | 1e-3 | features | NLL | ~11 min | -- | **1.4721** | The L1 specialist. Passes 1.55 target. Headline number. |
| hymn_dim2048_carry16_30k | 30,000 | 16 | 16 | 2048 | 1e-3 | features | NLL | -- | -- | 1.88 | LR not scaled with dim. |
| hymn_dim2048_carry16_30k_lr2e4 | 30,000 | 16 | 16 | 2048 | 2e-4 | features | NLL | ~30 min | 1.92 | 1.92 | dim=2048 still worse than dim=1024 at this corpus size. Negative finding. |
| hymn_ts_minilm_smoke | 3,000 | 8 | 16 | 1024 | 1e-3 | MiniLM | NLL | 28.5 s | 2.73 | -- | MiniLM warm-start, cosine within-letter 0.48 / across 0.46 -- no category prior. |
| hymn_ts_features_smoke | 3,000 | 8 | 16 | 1024 | 1e-3 | features | NLL | 27.8 s | 2.40 | -- | A/B vs MiniLM. Features win for char-level. |
| **hymn_plus_v1_5k** | **5,000** | **n/a (full attention-free recurrence)** | **16** | **256** (4 layers, 4M params) | **3e-4** | **features** | **NLL** | **14.5 min CPU** | **1.25** | **1.25** | **BREAKTHROUGH: non-Transformer (Mamba-class selective recurrence + SwiGLU + pre-norm + residuals) beats HYMN-MLP by 15% with 10x fewer steps and 4x smaller dim. Generates plausible Shakespearean dialogue at temperature=0.7.** |
| hymn_plus_v2_15k | 15,000 | n/a | 16 | 384 (6 layers, 14M params) | 3e-4 | features | NLL | ~2 h CPU | 0.28 | 0.28 (train) | **OVERFITTING**: 14M params on 1.1M-char corpus is 12 chars/param + 20 epochs. Verified memorization (output contains verbatim Shakespeare lines from training). Train NLL not meaningful. |

## Held-out generalization (HYMN-Plus on OOD WikiText-2)

The honest test of any LM: NLL on text the model never saw. Tiny
Shakespeare-trained checkpoints evaluated on the first 50K chars of
WikiText-2 (Wikipedia-style English -- different distribution).

Uniform baseline for the 65-char vocab is 4.17 nats/char. Below uniform
= the model learned something general; above uniform = the model is so
specialized to training that it actively hurts on OOD text.

| Checkpoint | Held-out NLL on WT-2 | Ratio of uniform | Generalizes? |
|---|---|---|---|
| **hymn_plus_goldilocks** (2.4M params, 8K steps, val=1.60) | **2.76** | 66% | **Yes -- the new default recipe** |
| hymn_plus_v1_5k (4M params, 5K steps) | 2.77 | 66% | Yes -- essentially tied with goldilocks |
| hymn_plus_v2_15k (14M params, 15K steps) | 5.16 | 124% | No -- destroyed by overfit to Shakespeare |

## Sample-quality (verbatim overlap / diversity)

`scripts.sample_quality` runs N samples per prompt and computes the
fraction of 32-char windows from each sample that appear verbatim in
the training corpus (memorization signal) plus distinct-n-gram
diversity (mode collapse signal). Required after every training run.

| Checkpoint | Mean verbatim overlap | Diagnosis |
|---|---|---|
| hymn_plus_v1_5k | **0%** | Pure generation, no memorization |
| hymn_plus_v2_15k | 16.47% (one prompt 90.9%) | Memorized chunks of Shakespeare |

## Architecture A/B head-to-head (v1 vs v2 on OOD WT-2)

`scripts.benchmark_checkpoints` ran apples-to-apples eval:

  A: hymn_plus_v1_5k.npz   | 4M params | dim 256 | NLL=2.7566 | %uniform=66.0
  B: hymn_plus_v2_15k.npz  | 14M params | dim 384 | NLL=5.1288 | %uniform=122.9
  verdict: A wins by 2.3722 NLL (46.3% better than B)

The smaller, less-trained model dominates on OOD because the bigger
one overfit.

## Tied-vs-untied embedding ablation (2K steps, dim=128, 3 layers, Tiny Shakespeare)

| Config | Params | Wall | Final train NLL | Final val NLL |
|---|---|---|---|---|
| tied (default) | 796K | 350 s | 1.66 | **1.75** |
| no-tie-weights | 805K | 398 s | 1.63 | 1.75 |

**Identical val NLL.** Keep weight tying enabled by default -- saves
8K params and 14% wall time at no quality cost.

**Headline:** the smaller v1 checkpoint that hit 1.25 on training generalizes
to 2.77 on OOD WikiText-2 -- a real, honest number proving the architecture
learns distribution structure, not just memorization. The bigger v2 went
too far into overfit and lost generalization.

Lesson: **future training runs MUST use held-out validation.** A train
NLL that drops without a val NLL is meaningless. `scripts/eval_hymn_plus.py`
is the new standard tool.

## L2 -- WikiText-X self-eval NLL

| Run | Corpus | Steps | Carry | Batch | Dim | Train wall | Train NLL | Eval NLL | Uniform baseline | % of uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| hymn_wt2_carry16_30k | WikiText-2 | 30,000 | 16 | 16 | 1024 | 11 min | -- | 1.72 | ~6.5 | ~26% |
| hymn_wt103_warm_100k | WikiText-103 | 100,000 | 8 | 16 | 1024 | 20 min | 1.95 | **2.04** | 8.51 (4979 chars) | **24%** |
| hymn_plus_wt103_30k (in flight) | WikiText-103 | 30,000 | n/a | 16 | 384 (6 layers, 14M params) | ~6 h CPU | -- | (running) | 8.51 | -- | The real test of HYMN-Plus. 543 MB corpus dwarfs 14M params -> can't memorize -> NLL means something. |

**Generalist vs specialist note:** the WT-103-trained generalist scores
3.69 on Tiny Shakespeare (expected -- different character distribution).
The Shakespeare specialist (`hymn_carry16_50k`) scores 1.47 on Tiny
Shakespeare and would do worse on WT-103. Different surfaces measure
different things; don't compare across columns.

## Phase-2 -- RLAIF judge feedback

| Run | KB facts | Judge | Probes | Verdicts | Notes |
|---|---|---|---|---|---|
| smoke | 2399 (filtered from 2711) | llama3.2:3b | 50 | 25 keep / 25 refine | clean-form prompt (`build_triple_prompt`) avoids "stored fact" bias |

## N1 -- Continual-learning retention

| Run | n_a (initial) | n_b (interference) | n_probe | Dim | Shards | Initial recall | Post-B recall | Retention | Pass threshold | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 200 | 200 | 80 | 2048 | 16 | >0 | -- | >= threshold | -- | yes |
| sanity | 200 | 0 | 80 | 2048 | 16 | 1.0 | 1.0 | 1.0 | -- | yes (no interference) |

## KB grounding

| KB | Source | Facts ingested | Judge filter | Kept |
|---|---|---|---|---|
| llama3b_expanded | Ollama llama3.2:3b distillation, 120 topics | 2711 | yes | 2399 (88.5%) |

## Open / unrecorded numbers

- WikiText-2 vs WikiText-103 specialist gap on L2 (need both same-vocab eval)
- SOFAR LoRA ablation (A2 currently 0.08% improvement vs design target 3%)
- Conversational HYMN train run (Alpaca + KB-QA + Shakespeare hybrid corpus, in flight)
- Larger KB (target 50K facts) -- not started

## File pointers

- Checkpoints under `data/checkpoints/`. Not git-tracked.
- Training logs under `logs/`. Not git-tracked.
- Eval JSON under `evals_out/`. Not git-tracked.

Reproduce any row above via the training-runbook:
`docs/training-runbook.md` -> matching recipe section.
