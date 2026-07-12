> **Historical document (pre-pivot).** Projections and scaling plans in
> this file predate the 2026-05-27 project conclusion and pivot. For the
> current honest state see `docs/CONCLUSION.md` and `raincg/RESULTS.md`.

# RAIN-Net Training Strategy: Cheap-Compute Path to Competitive Capability

> The thesis: RAIN-Net's architecture lets us train a competitive
> generative AI for $1M-$10M instead of $100M+, using distillation,
> active learning, and synthetic data. This document is the concrete
> recipe.

## Why this works (the architectural argument)

LLMs pay massive compute because they compress the entire internet into
weights. That compression IS the model: the weights have to encode every
fact, every skill, every reasoning pattern. Pretraining cost scales with
the corpus size and the parameter count.

RAIN-Net does not compress the internet into weights. The base sequence
model only learns to compose hypervectors. Facts live in addressable
memory. Skills live in compiled procedures. Multi-modal capability lives
in modality-specific encoders that all produce HVs in the shared space.
The base model's job is much narrower: be a competent sequence
encoder/decoder + composition engine.

This narrowness is what makes the small-compute path viable. The base
model doesn't need to know who won the 1996 World Series; it just needs
to know how to retrieve that fact from memory and produce fluent text
around it.

## Five training mechanisms (combined)

### 1. Teacher-LLM distillation (the cold start)

We do NOT pretrain on raw web text. We distill from a frontier LLM as
teacher. For each query in the curriculum:

1. Send query to teacher (Ollama Qwen 7B locally, or Claude API).
2. Get teacher's answer + citations + reasoning trace.
3. Train student via KL(student || teacher) + KB-grounding loss.

The student inherits the teacher's knowledge compression at fraction
of the compute. **This is how Phi-3, Qwen-0.5B-3B, Llama-3.2-1B were
all built** — small-model state of the art comes from distillation.

What we add on top:
- **KB-grounding loss**: the student must produce the teacher's answer
  *only* when the relevant fact is injected via KB-Attention. This
  forces the student to use the addressable memory as the source of
  truth rather than memorising into weights.
- **Multi-modal joint distillation**: text + code + image-caption
  teachers all distill into the same HV-substrate student, training
  the encoder bank to produce consistent HVs across modalities.

Compute estimate:
- 300M base + 1B distillation tokens: ~$1,500-$3,000 RunPod H100-week
- OR 2-4 weeks on the Corsair AI Workstation 300 (Ryzen AI MAX+ 395 +
  Radeon 8060S iGPU via DirectML + 128GB unified RAM): $0
- Versus $1M+ for from-scratch pretrain to similar capability

### 2. Active distillation (lifelong, free)

After cold start, the model improves during use. Every query:

```
user_query → RainNet.answer()
            ├─ confidence ≥ threshold → return answer (free, local)
            └─ confidence < threshold:
                ├─ ask Ollama teacher (free, slower)
                ├─ ingest teacher's answer as new semantic fact
                ├─ save as distillation training example
                ├─ if still low confidence + Claude opt-in:
                │   └─ escalate to Claude (cost-gated)
                └─ return teacher's answer to user
```

The marginal cost per query **asymptotes toward zero** as the KB grows.
More queries answered from local memory; fewer trigger teacher
escalation. Implemented in `rain/training/active_learning.py`.

This is the genuinely novel commercial pattern. The model trains itself
during use. It's enabled by the architectural separation between frozen
base weights and growable addressable memory.

### 3. Synthetic-data curriculum (quality over quantity)

Microsoft's *Textbooks Are All You Need* (Gunasekar 2023) showed
1-7B-token high-quality synthetic data beats 100B+ raw web tokens at
matched parameter count. We use the same approach.

For each target domain, we generate a curriculum of
(question, scratchpad reasoning, KB citations, answer) quadruples via
the teacher LLM. The student trains on this distilled curriculum.
Quality is enforced by:
- Filtering teacher answers via a second teacher (LLM-as-judge)
- Pruning examples where citations don't support the answer (binding
  coherence check)
- Deduplicating by HV similarity

Implemented in `rain/training/distillation.py::synthetic_curriculum_queries()`.

### 4. Local-first scaling on consumer hardware

The workstation tier (no cloud cost):
- Corsair AI Workstation 300 (Ryzen AI MAX+ 395, Radeon 8060S iGPU via
  DirectML, 128GB unified RAM) can train 300M-1B parameter models with
  patience.
- DirectML support means AMD iGPU is usable without CUDA dependency.
- 128GB unified means batch sizes up to 32 at 512-token sequences.
- 2-4 weeks per training run is acceptable for a research iteration cadence.

What gets done on workstation:
- Cold-start distillation at 300M (this completes in ~2 weeks).
- Active-learning fine-tuning loops (incremental, hours per cycle).
- All evaluation runs (apples-to-apples vs Pythia-160M, Mamba-130M).
- Synthetic curriculum generation (CPU-bound, runs in parallel).

What needs cloud:
- 1B+ scale runs (workstation memory becomes the bottleneck).
- Multi-modal joint distillation (image teachers want GPU memory).
- Inference benchmarks at production batch sizes.

### 5. Frozen-base, addressable-memory continual updates

Once the base is trained, **it is frozen forever**. No more gradient
updates to the base weights. New knowledge enters via:

- **Semantic memory adds** (free, instant): KB-Attention sees them next query
- **Procedural skill adds** (small LoRA-style adapter, ~10MB, trained
  in minutes on workstation, keyed by HV pattern)
- **Modality adds** (new encoder, no base change)
- **Verifier updates** (online perceptron updates, free)
- **Router domain HV nudges** (online vector updates, free)

This is the structural answer to catastrophic forgetting. There are no
weight updates to forget.

## Cost tiers (concrete budgets)

| Tier | Compute | Time | Outcome |
|---|---|---|---|
| **Tier 0** | $0 (workstation only) | 2-4 weeks | Architecture proof: 300M base distilled from local Ollama. KB seeded with ~50K facts. v0.1 architectural paper ready. **CURRENT TIER.** |
| **Tier 1** | $5K (RunPod H100, 1-2 weeks) | 3-4 weeks | 300M base distilled from Claude API. Multi-modal v0.1 paper submitted to NeurIPS/ICML workshop. Working demo against Phi-3 Mini on continual benchmarks. |
| **Tier 2** | $50K (8×H100 1 month) | 2-3 months | 1B base. Full MoA bank with each expert at 100-200M. Beats Mamba-130M / Pythia-160M on standard LM benchmarks and demolishes them on continual + audit benchmarks. ICML/NeurIPS main-track-quality paper. |
| **Tier 3** | $500K (cluster, 3 months) | 6 months | 3B base. Production-deployable narrow vertical (legal/aviation/medical) wins. First paying customers. Pre-seed extension or Series A ready. |
| **Tier 4 (Series A target)** | $5M | 6-12 months | 7B base, full multi-modal training, evaluation harness covering all benchmarks. Frontier-narrow competitive on vertical capabilities. 5-10 ML engineers + infra team. **This is what Series A funds.** |
| **Tier 5 (Series B)** | $50M+ | 12-24 months | GPT-class general capability via composition. Multi-vertical product. Independent benchmarks vs frontier LLMs. |

## Cost comparison vs LLM pretraining at matched capability

| Capability tier | LLM training cost | RAIN-Net training cost | Saving |
|---|---|---|---|
| ~Llama-3.2-1B level | $500K-$2M (pretrain) | $5K-$50K (distill + KB) | 10-400x |
| ~Llama-3.1-8B level | $5M-$20M (pretrain) | $50K-$500K (distill + KB + MoA) | 10-100x |
| ~Llama-3.1-70B level | $50M-$200M (pretrain) | $1M-$10M (full stack) | 5-50x |
| ~GPT-4 level | $100M+ | $10M-$50M (full stack + scale) | 2-10x |

These numbers assume distillation cap (student cannot exceed teacher).
The savings shrink at the top because at GPT-4 capability the teacher
itself is expensive to run, and the marginal scaling is harder. But
even at the top tier, the cost gap remains an order of magnitude.

## Inference cost comparison (where the larger win is)

LLMs pay for inference proportional to model size. A query to GPT-4
costs ~$30/M input tokens, $60/M output tokens.

RAIN-Net routes to a small base (300M-3B params) plus the top-k experts
(say 2 of 8, ~200M each). Total active params per query: ~700M-1B.
On a workstation: free. On rented cloud GPU: ~$0.50/M tokens.

For a startup serving customer queries, this is the difference between
$30/customer/day and $0.50/customer/day — a 60x margin advantage.
Combined with the continual-learning advantage (no need to retrain
for new content), RAIN-Net's unit economics are structurally better.

## Investor narrative summary

> "We're training a competitive generative AI for 10-100x less than
> what frontier labs spend, using teacher distillation + addressable
> memory + active learning. The base model is small (1-7B); the
> intelligence is in the composition + memory + verifier loop. Once
> trained, we don't pay to update — facts get added to memory, skills
> get compiled to small adapters, the base stays frozen. Inference
> costs are 50-100x less than GPT-class. Continual learning + audit
> trail are structural — properties competitors can't match by tuning
> hyperparameters."

That is the Series A pitch in one paragraph.
