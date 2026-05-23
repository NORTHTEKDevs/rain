# HYMN Training Runbook

> Empirical hyperparameters and gotchas collected while training HYMN on the
> Corsair AI Workstation 300 (Ryzen AI MAX+ 395, Radeon 8060S, 128 GB DDR5
> unified). Last updated: 2026-05-22.

## Quick decision tree

| Question | Answer |
|---|---|
| Which trainer? | `scripts.pretrain_hymn_torch` (Adam-W + batching + DirectML). Numpy reference at `scripts.pretrain_hymn` stays as the deterministic checkpoint-format anchor only. |
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
