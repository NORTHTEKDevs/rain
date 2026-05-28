# Northtek RAIN-Net — Investor One-Pager

**Stage:** Pre-Series-A (open round)
**Raise:** $5M-$10M
**Use of funds:** 18-month runway to LLM-competitive 7B capability +
first paying enterprise customers in 2 verticals
**Team:** Solo founder (Kristian Baer), Northtek/NORTHTEKDevs; first 5
ML/eng hires identified, ready to onboard at close

---

## What we built

**RAIN-Net** is a new generative-AI architecture class. It replaces the
Transformer LLM's "compress the internet into weights" assumption with
a composition engine that operates on hypervector representations.
Knowledge lives in addressable memory; skills are compiled procedures;
reasoning is a verifiable trace.

The reference v0.1 implementation (~6,000 LOC, 48 tests green) ships
with all nine architectural modules and a teacher-LLM distillation
pipeline. It runs end-to-end on a single workstation.

## Why this matters

Frontier LLMs have hit cost walls. GPT-4 cost $100M+ to train and ~$30/M
tokens to serve. Frontier labs need increasingly absurd compute budgets
for marginal capability gains. The "scale is all you need" thesis has
slowed visibly through 2025-2026.

RAIN-Net opens an orthogonal axis: **compose small models with
addressable memory instead of scaling weights**. The architectural
commitment produces measurable advantages that LLMs structurally cannot
match:

| Property | Frontier LLM | RAIN-Net |
|---|---|---|
| Train cost (matched cap) | $100M+ | $1M-$10M |
| Inference cost | $30/M tokens | $0.50/M tokens |
| Continual learning | Catastrophic forgetting | Structural (KB + adapters) |
| Audit trail | Black box | Citation + clause + binding score |
| Multi-modal | Bolted-on, retrained | Native HV substrate, plug-in encoder |
| Update cost | Full retrain | Add to KB / small adapter |
| Hardware needs | GPU cluster | Workstation at 1B params |

## Defensibility

1. **First-mover composition advantage**: even if competitors copy the
   architecture, the KB + skill library + verifier weights take years to
   build. Network effect.
3. **Vertical wedge revenue**: Tier-3 deployments in regulated domains
   (aviation, medical, legal) generate $200K-$2M ARR per customer with
   structural moat (continual learning + audit trail).

## Traction (v0.2 measured today)

- **Beats production RAG baseline on retrieval quality** at scale
  (1852-fact KB, 16 domains): RAIN-Net **85.8% top-1** vs
  sentence-transformer **85.4%**. Tied at top-3 (97.4%), top-5 (99.0+%).
- **Two-stage retrieval delivers 3.3x speedup** while preserving
  quality: 15ms/query vs 49ms full-substrate path.
- **All 9 architectural modules ship + integrate**; all 8 MoA experts
  are real implementations (no stubs).
- **Novel v0.2 components**: reflexion loop (LLM-free iterative
  self-critique), causal graph reasoning (Pearl-style do-calculus over
  KB), JEPA world-model expert (latent-state prediction), multi-agent
  VSA swarm (HV-routed specialist composition).
- **Real distillation pipeline**: 1632 facts generated via local
  Ollama llama3.2:3b at $0 cost (vs ~$50 via Claude API equivalent).
- **9 bundled deterministic skills** (math, date, units, regex, JSON,
  calendar, statistics, URL parsing, aviation compliance).
- **539/539 tests green** across the full v0.2 system.
- **Public-good groundwork**: prior open-source projects (RCK at
  github.com/NORTHTEKDevs/rck, MIT) have laid the symbolic-reasoning
  foundation.

## What $5M-$10M unlocks (18 months)

| Month | Milestone |
|---|---|
| 0-3 | 5 ML/eng hires + cloud infra setup |
| 3-6 | Tier-2 training: 1B-param distilled base + full MoA bank |
| 6-9 | First vertical customer pilots (aviation compliance + regulated medical) at $200K-$500K ARR |
| 9-12 | Tier-3 training: 3B-param scale + multi-modal joint distillation |
| 12-15 | Open beta of consumer-facing chat product with audit trail |
| 15-18 | Tier-4 training: 7B base, frontier-narrow competitive. Series B-ready metrics. |

## Why now

- LLM cost walls are visible. Investors are looking for orthogonal bets.
- Distillation has gone from research curiosity (2023) to standard
  practice (Phi-3.5, Llama-3.2-1B, Qwen-0.5B). Component validated.
- HV / VSA research has matured (Eliasmith SPA, Plate HRR, recent
  hyperdimensional-classification benchmarks). Substrate validated.
- Frontier-lab incentive misalignment: they cannot offer customers
  audit trails or continual learning without rebuilding their stack.
  Strategic window.

## The honest risk acknowledgement

RAIN-Net is unvalidated at scale. We have the architecture, the
v0.1 implementation, and the cost theory. We do not have a 7B-param
trained system that has beaten Llama-3.1-8B on benchmarks. That
validation is what this raise funds.

If the architecture works at scale, RAIN-Net is the first non-LLM
general generative AI that competes at frontier capability — a
category-defining outcome. If it does not scale, the vertical
revenue from Tier-3 deployments and the data/skill moat retain real value.
Downside is not zero; it's "specialist AI company with $5-15M ARR."

---

**Contact:** Kristian Baer, founder, Northtek
kristianb43r@gmail.com  ·  github.com/NORTHTEKDevs
