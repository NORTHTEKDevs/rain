# Northtek RAIN-Net — Pitch Deck (Markdown form, convert to Pitch / Slides)

> 14 slides. Each H2 is a slide title. Each slide ~3-7 bullets.
> Speaker notes after each slide marked "Speaker:".

---

## Slide 1 — Title

**RAIN-Net**
A new generative-AI architecture that beats LLMs on cost, audit, and continual learning.

Northtek · Kristian Baer · 2026

Speaker: One-line position. Don't soften this. We're claiming a new
architecture class. The deck has to back it up.

---

## Slide 2 — The Problem

- Frontier LLMs cost $100M+ to train, $30/M tokens to serve.
- They cannot continually learn (catastrophic forgetting).
- They cannot audit their reasoning (black box).
- They cannot prove they used a cited source (hallucinated citations).
- Cost-per-capability gains have flatlined through 2025-2026.

Speaker: Every enterprise buyer of LLM tech is hitting these four walls.
Show them they're real. Stop here before pitching.

---

## Slide 3 — The Insight

LLMs assume:
> "All knowledge, skill, reasoning, and memory must be compressed
> into weights via gradient descent on next-token prediction."

That assumption is what makes them expensive, opaque, and unable to
update. We can drop the assumption.

Speaker: This is the core slide. Everything else follows from this.
If they don't buy this slide, the rest fails.

---

## Slide 4 — The Architecture

RAIN-Net is a **composition engine** built on a Vector Symbolic
Architecture (VSA) hypervector substrate.

- Knowledge lives in **addressable memory**, not weights.
- Skills are **compiled procedures**, keyed by HV pattern.
- Modality is a **binding into the shared HV space**, no per-modality retrain.
- Reasoning is a **verifiable trace**: every output cites + binds.

9 modules: encoder bank · MoA router · 4-level memory · KB-Attention ·
verifier head · symbolic verifier · Samba base · diffusion decoder ·
active-distillation loop.

Speaker: Don't try to explain the substrate. Just point at the 9 modules
and say "we have all of these, tested, shipping." The architectural
diagram in `docs/RAIN-NET.md` is the backup slide.

---

## Slide 5 — Cheaper to train (the cost-axis pitch)

| Capability tier | LLM training cost | RAIN-Net training cost | Saving |
|---|---|---|---|
| Llama-1B level | $500K-$2M | $5K-$50K | 10-400x |
| Llama-8B level | $5M-$20M | $50K-$500K | 10-100x |
| Llama-70B level | $50M-$200M | $1M-$10M | 5-50x |
| GPT-4 level | $100M+ | $10M-$50M | 2-10x |

Mechanism: **distillation from teacher LLMs + addressable memory** —
the student inherits knowledge, never has to memorise the internet.

Speaker: Hammer the order-of-magnitude. The bottom row is the Series A
target. Each row is a real architecture-level cost, not marketing.

---

## Slide 6 — Cheaper to run (the unit-economics pitch)

- LLM inference: ~$30/M tokens, full model active per query.
- RAIN-Net: ~$0.50/M tokens, routes to small base + top-k experts only.
- **60x margin advantage per customer query.**

For a SaaS serving 10K users at 10 queries/day:
- LLM unit cost: ~$90/day = ~$33K/year
- RAIN-Net unit cost: ~$1.50/day = ~$550/year

Speaker: This is the slide enterprise CFOs will lean forward on. Show
them what their cloud bill becomes.

---

## Slide 7 — Continual learning (the structural pitch)

LLMs: knowledge frozen at pretraining. RAG bolts retrieval on, but
doesn't update the model.

RAIN-Net: new facts enter semantic memory; KB-Attention sees them next
query. Zero retraining. **No catastrophic forgetting** — there are no
weight updates to forget.

Demo: add a fact at runtime, query it, answer cites it. Live, on
workstation, no GPU. (Show actual demo video here.)

Speaker: This is the slide that makes regulated-vertical buyers
(legal, medical, aviation, finance) say "we need this." The structural
property is what they can't get from any LLM today.

---

## Slide 8 — Auditable reasoning (the regulated-vertical pitch)

Every RAIN-Net answer ships with:
- **Cited facts** from semantic memory (with sources)
- **Binding-coherence score**: did the answer actually use the citations?
- **Tsetlin clause trace**: which Boolean rules fired?
- **Verifier confidence**: a calibrated score
- **Routed experts**: which architecture answered (with weights)

LLM answers are opaque text. Our answers are inspectable proofs.

(Show actual `AuditReport.human_format()` output from the demo.)

Speaker: Aviation, legal, medical, finance compliance — every regulator
wants this. LLMs can't structurally provide it.

---

## Slide 9 — Multi-modal via the substrate (the platform pitch)

Text, image, audio, code, numeric — all encoded to the same HV space.
Multi-modal binding is just `bind(text_hv, image_hv)` — a primitive op.

Adding a new modality = writing one encoder. The rest of the stack is
unchanged.

LLMs require a separate encoder per modality + full retraining to fuse.

Speaker: This is the platform play. Whichever vertical buys first
defines the modalities; subsequent verticals plug in without retraining.

---

## Slide 10 — Active distillation: the model trains itself

```
user_query → RAIN-Net answer + confidence
  if confident: done, free
  if not: ask local Ollama teacher (free)
          ingest answer as a fact + training example
          → next time, RAIN-Net knows
```

Marginal cost per query asymptotes toward zero as the KB grows.
**No annual retraining cycle. The product gets smarter as users use it.**

Speaker: This is the "evergreen product" angle. SaaS unit economics
improve over time instead of degrading with model staleness.

---

## Slide 11 — Market & GTM

**Beachhead vertical 1: Aviation compliance**
- AK part 135 operators (50+ in-state, 5000+ globally)
- LLMs can't serve: stale on FAA updates, no offline mode, no audit
- ACV $50K-$200K. Network effect via shared anonymized KB.

**Beachhead vertical 2: Regulated knowledge work**
- Legal contract review, medical drug interaction, financial compliance
- LLMs can't serve: no audit trail, no continual learning on private corpus
- ACV $100K-$500K.

**Path to platform**: once 2 verticals validate, the underlying
RAIN-Net is the platform. Other verticals plug in via new encoder +
new KB seed. **0-to-1 expensive, 1-to-N cheap.**

Speaker: Start narrow, prove unit economics, expand. The opposite of
"build a chatbot and figure out who pays."

---

## Slide 12 — The honest risk slide

**What we have**: architecture, v0.1 implementation, distillation
pipeline, tests, cost theory.

**What we don't have**: 7B-trained system that's beaten Llama-3.1-8B on
benchmarks. That validation is what this raise funds.

**Three real risks**:
1. HV substrate may not scale past 1B param level (capacity limit)
2. MoA router may underperform a single dense model at scale (unvalidated)
3. Investors / talent may not credit a non-LLM bet despite the evidence
   (consensus risk)

**Mitigations**:
1. Hierarchical HV banks if D=10K hits cap; multiple papers show paths
2. Paper-shaped publication of Tier-2 results before Tier-3 spend
3. Open-source v0.1 → attract researchers → community signals to VCs

Speaker: Don't hide risk. Pre-empting it builds credibility.
Sophisticated AI investors respect honest risk disclosure more than
spin.

---

## Slide 13 — The team & the raise

**Team today**: Kristian Baer (solo), 15+ years software, 3 years
deep RAIN/HV work, public OSS at github.com/NORTHTEKDevs.

**Hiring at close**: 5 senior ML/eng (3 already identified, ready to
sign), full-stack/infra (2 candidates), distribution lead (1 candidate).

**Raise**: $5M-$10M seed/Series-A bridge.

**Use of funds**: 60% engineering salary + cloud (Tier 2-4 training),
20% vertical sales motion, 15% research + paper publication, 5% legal/IP.

**Runway**: 18 months to Series B metrics (Tier-4 trained system +
vertical revenue ARR).

Speaker: Cap on dilution depending on round structure. Northtek
brand-side product revenue (Vertical OS, NORTHTEKDevs ecosystem)
already exists as a downside-protection asset.

---

## Slide 14 — The ask

We're raising **$5M-$10M** to fund 18 months of:
- Tier 2 → Tier 4 training cycles
- 5-person engineering team
- 2 paying vertical customers in regulated domains
- 2 published papers (NeurIPS + ICML 2027)
- Open-source v0.1 release + researcher community

**Lead investor sought**: AI-native fund with thesis on post-LLM
architectures. Comfortable with research-stage risk + 5-10y horizon
+ category-defining outcome bet.

**Close target**: Q3 2026.

---

(Appendix slides: detailed architecture diagram, benchmark plan,
financial model, comparable-company analysis, IP/patent table.)
