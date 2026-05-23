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

## L2 -- WikiText-X self-eval NLL

| Run | Corpus | Steps | Carry | Batch | Dim | Train wall | Train NLL | Eval NLL | Uniform baseline | % of uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| hymn_wt2_carry16_30k | WikiText-2 | 30,000 | 16 | 16 | 1024 | 11 min | -- | 1.72 | ~6.5 | ~26% |
| hymn_wt103_warm_100k | WikiText-103 | 100,000 | 8 | 16 | 1024 | 20 min | 1.95 | **2.04** | 8.51 (4979 chars) | **24%** |
| hymn_conversational_v1_60k | Hybrid (Alpaca 1x + KB-QA 3x + Shakespeare 1x, 20.8 MB) | 60,000 | 8 | 16 | 1024 | 10 min | 1.86 | (no held-out yet) | -- | -- |

## Conversational HYMN qualitative

`hymn_conversational_v1_60k` (NLL 1.86 on the training mix). Trained on
20.8 MB hybrid: 51,709 Alpaca Q/A pairs + 7,197 KB-derived Q/A pairs
(3x oversampled) + 7,222 Shakespeare paragraphs. Shuffled at chunk level.

**Observed at 60K steps**:
- Always emits the Q:/A: alternation pattern (structure learned)
- Periods after sentences, blank lines between Q/A blocks (formatting learned)
- Short common words (and, is, as, the, A:) reproduced correctly
- Longer content reads as gibberish -- expected at char-level for 60K steps

**Honest assessment**: HYMN learned the SURFACE FORM of Q/A dialogue but
not yet the SEMANTICS. This matches RAIN's design thesis: HYMN is the
sequence engine (form/fluency), KB is the knowledge source (semantics).
The combined system in agent.ask is what produces grounded answers.

**Next training run to push semantics**: re-train at 300K+ steps,
ideally larger hidden_dim, ideally with conversational data >100 MB.
Not blocking v0 demo.

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
