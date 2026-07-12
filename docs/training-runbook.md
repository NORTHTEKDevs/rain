# HYMN / HYMN-Plus Training Runbook

> Empirical hyperparameters and gotchas. Two architectures live here now:
> **HYMN** (legacy 2-layer MLP with carry-step) and **HYMN-Plus**
> (selective gated recurrence + SwiGLU + pre-norm + residuals, the
> non-Transformer LM that actually generates text).
>
> Corsair AI Workstation 300 (Ryzen AI MAX+ 395, Radeon 8060S, 128 GB
> DDR5 unified). Last updated: 2026-05-23.

## Quick decision tree

| Question | Answer |
|---|---|
| Which architecture? | **HYMN-Plus** for any new run -- beats HYMN by 15%+ at smaller param count, generates real text. Use HYMN only for backward-compat checkpoint reads. |
| Which trainer? | `scripts.pretrain_hymn_plus` for HYMN-Plus. `scripts.pretrain_hymn_torch` for HYMN. |
| Which loss? | `--loss nll` for any new run. MSE-on-HV is the v0 reference and trained slower per signal-bit. |
| Which device? | `--device directml` on this workstation. CPU is competitive at dim ≤ 1024 but DirectML wins as dim grows. |
| Which corpus? | Start with Tiny Shakespeare (~1.1 MB, in `data/corpora/`). Bump to WikiText-2 when train and val both look healthy. |

## Tuned defaults (per loss type)

### Loss = MSE-on-HV (the v0 reference)
- `--lr 1e-3`
- `--weight-decay 0`
- `--batch-size 64`
- `--context-len 8`
- `--in-dim 1024 --hidden-dim 512 --out-dim 1024`
- Diverges only at LR > ~2e-3.
- Best L1 HV-MSE observed: **0.86 - 0.89** on Tiny Shakespeare. Plateaus around
  50K steps regardless of more compute -- corpus + loss are the ceiling, not steps.

### Loss = NLL (real cross-entropy)
- `--lr 5e-4`  (CRITICAL: lr=2e-3 diverges after ~step 2000; loss climbs from 3.8 back to 5.2.)
- `--weight-decay 1e-4`
- `--batch-size 64`
- `--context-len >= 8` (16 better for longer runs)
- `--in-dim 1024 --hidden-dim 512 --out-dim 1024`
- Initial NLL on random init: ~27 nats/char (much higher than uniform 4.17 because
  the HYMN output is concentrated in wrong directions before any training).
- After 10K steps: train ~2.7, L1 val ~3.4.
- Design-plan target: 1.55 nats/char (uniform-random baseline is 4.17 on
  Tiny Shakespeare's 65-char vocab).

### Loss = NLL with sequence-carry (RNN-style, the one that PASSES L1)
**This is the recipe that crossed the 1.55 design-plan threshold.**
- `--loss nll`
- `--carry-steps 16` (W=16 chars of teacher-forced state evolution)
- `--grad-clip 1.0` (REQUIRED at W >= 4 to prevent BPTT explosion)
- `--lr 5e-4`
- `--weight-decay 1e-4`
- `--batch-size 64`
- `--in-dim 1024 --hidden-dim 512 --out-dim 1024`
- L1 eval picks `eval_carry_steps=16` automatically from the checkpoint sidecar.

Measured results on Tiny Shakespeare:
| carry | steps | wall | train (last 2K) | L1 val NLL |
|---|---|---|---|---|
|  0 | 100K | 559s CPU | 2.70 | 2.92 |
|  4 |   5K |  48s CPU | 2.47 | 2.34 |
|  8 |  30K | 483s CPU | 1.84 | 1.72 |
| 16 |  30K | 833s CPU | 1.64 | **1.5359**  pass=true |

Cumulative reduction: 47% in one shift. The lever is sequence-carry training,
not the architecture or more steps.

Note on the bootstrap bug to avoid: the naive RNN loop that decodes the raw
codebook vector at step 1 (before any HYMN forward) produces loss ~250 nats/char
because the codebook vector dot-products to ITSELF with overwhelming preference.
The correct order is `state = HYMN(state, codebook(char_t)); decode(state)` --
HYMN before decode at every step, including the first. See
`train_torch.carry_steps>0` path.

## Throughput (this workstation, dim=1024, batch=64, context=8)

| Backend | Steps / sec | Wall for 50K |
|---|---|---|
| CPU (Zen 5, 32 threads) | ~203 | ~250 s |
| DirectML (Radeon 8060S) | ~154 | ~325 s |
| numpy reference (CPU, single-example) | ~10 | ~5000 s |

DirectML wins on raw matmul (6x on 2048x2048 fp32) but loses some of that to:
- `aten::lerp.Scalar_out` fallback to CPU inside Adam/AdamW.
- Per-step batch transfer host -> iGPU.

Threshold to prefer DirectML: dim ≥ 2048 OR batch ≥ 128 OR sequence-carry across
multiple HYMN forwards per step.

## Two-job rule

Do NOT run two simultaneous DirectML jobs on this workstation. The torch-directml
plugin can produce a `device.index() < device_ready_queues_.size()` INTERNAL ASSERT
when contended. If a long DirectML training is in-flight, queue secondary jobs to
`--device cpu` instead.

Concurrent Ollama inference (e.g., `seed_kb_from_ollama` running qwen30B) plus a
DirectML HYMN training is OK because Ollama uses unified RAM differently than
torch-directml's device queue.

## Known divergence signatures

| Loss type | Symptom | Fix |
|---|---|---|
| NLL | Loss falls fast, then climbs after a plateau | Lower LR (target 5e-4 to 3e-4), enable weight_decay |
| Both | Loss NaN / Inf in first few steps | Codebook vector contains a non-finite value (bug, not hyperparam); reseed Codebook |
| MSE | Loss flat at ~1.0 forever | Underfit. Check that `input_` is not always zero when `context_len > 0` was requested |
| NLL | Val loss 100+ nats/char on a 65-char vocab | Wrong checkpoint -- the checkpoint was trained with MSE, can't be evaluated with NLL meaningfully |

## L1 (Tier-2 LLM-parity, Tiny Shakespeare)

Run with:
```bash
python -m evals.tier2_llm_parity.tiny_shakespeare \
    --checkpoint data/checkpoints/<your_ckpt>.npz \
    --corpus data/corpora/tiny_shakespeare.txt \
    --metric nll \
    --n-eval-chars 5000 \
    --out evals/results/$(date +%F)/L1.json
```

`--metric nll` reports nats/char against the 1.55 design-plan target.
`--metric hv_mse` reports HV-MSE against the v0 0.5 proxy.

## What's next (not yet shipped)

- Cosine-decay LR schedule with warmup. Helps NLL stability at higher LR for the
  first few thousand steps.
- Sequence-carry training: instead of independent next-char positions, evolve the
  HYMN state across a window and backprop through the sequence. Closer to the
  inference-time semantics.
- WikiText-2 pretrain. Much more data per epoch; should close the train/val gap
  visible at 10K steps on Tiny Shakespeare.
- Real NLL eval on a checkpoint trained with NLL + 50K steps + context=16 + weight_decay.
  Latest in-flight at the time of writing.

---

# HYMN-Plus Training Recipes (the new default)

## Architecture

`rain.core.hymn_plus.HymnPlus`: N x (Selective-Gated-Recurrence + SwiGLU MLP)
with pre-norm + residuals + weight-tied output head. Non-Transformer. All ops
are matmul + elementwise. Bipolar codebook is used as the optional warm-start
prior for the learnable embedding table.

## Tuned defaults

### Goldilocks (the recommended starting point)

For Tiny Shakespeare or any small corpus where memorization is a risk:

```bash
python -m scripts.pretrain_hymn_plus \
    --corpus data/corpora/tiny_shakespeare.txt \
    --steps 8000 --batch-size 16 --seq-len 64 \
    --dim 192 --n-layers 4 --mlp-mult 4 \
    --lr 3e-4 --warmup-steps 400 --cosine-decay --weight-decay 0.05 \
    --grad-clip 1.0 --warm-start-chars \
    --val-split 0.05 --val-every 500 \
    --device cpu --log-every 500 \
    --out data/checkpoints/hymn_plus_goldilocks.npz
```

Why these numbers: 4M params on 1.1M-char corpus is ~4 chars/param, low
enough to learn distribution structure without dropping into pure memorization.
val-split required; if val gap grows large, stop early.

### Scale-up (when corpus dwarfs model)

For WikiText-103 (543 MB) or hybrid corpora >100 MB:

```bash
python -m scripts.pretrain_hymn_plus \
    --corpus data/corpora/wikitext103_train.txt \
    --steps 30000 --batch-size 16 --seq-len 96 \
    --dim 384 --n-layers 6 --mlp-mult 4 \
    --lr 3e-4 --warmup-steps 1000 --cosine-decay --weight-decay 0.05 \
    --grad-clip 1.0 --warm-start-chars \
    --val-split 0.02 --val-every 1000 \
    --device cpu --log-every 1000 \
    --out data/checkpoints/hymn_plus_wt103_30k.npz
```

14M params on 543 MB cannot memorize -> val NLL is the honest distribution
learning number.

## Honest validation rules (added 2026-05-23)

**Train NLL without val NLL is meaningless.** v2_15k hit train 0.28 but val
on out-of-distribution text was 5.16 (worse than uniform random) -- pure
memorization. Every new run MUST:

1. Use `--val-split` (default 0.05; raise to 0.1 for very small corpora)
2. Watch the val/train gap in the training log
3. After training, run `scripts.eval_hymn_plus --corpus <held-out>`
4. After training, run `scripts.sample_quality` -- verbatim_overlap > 5%
   means memorization

## Measured results (Tiny Shakespeare)

| Run | Steps | Dim | Layers | Params | Wall | Train NLL | OOD WT-2 NLL | Verbatim overlap |
|---|---|---|---|---|---|---|---|---|
| hymn_plus_v1_5k | 5,000 | 256 | 4 | 4M | 14.5 min CPU | 1.25 | 2.77 | 0% |
| hymn_plus_v2_15k | 15,000 | 384 | 6 | 14M | ~2h CPU | 0.28 | 5.16 (worse than uniform) | 16% (one prompt: 91%) |

**v1 wins.** v2 is documented as a negative result: too many params for the
corpus, memorized instead of learning.

## Sampling defaults

`temperature=0.7 top_k=30` are the v1 defaults that produced fluent Shakespeare
dialogue without descending into copy or into random gibberish. Lower
temperature gets more conservative output; higher gets more creative.

## Throughput (HYMN-Plus, CPU)

| Config | Steps / sec |
|---|---|
| dim=128, layers=2, seq=32 | ~37 |
| dim=192, layers=4, seq=64 | ~14 |
| dim=256, layers=4, seq=64 | ~6 |
| dim=384, layers=6, seq=96 | ~2 |

CPU dominates HYMN-Plus on this workstation -- the sequential Python loop
inside SelectiveGatedRecurrence kills DirectML throughput (per-op host-device
sync costs). Use `--device cpu` for HYMN-Plus until we ship a parallel scan kernel.
