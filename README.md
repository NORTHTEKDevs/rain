# RAIN — Resonant Active Inference Network

> A new class of generative AI that is not an LLM, not a state-space model, and
> not a transformer. RAIN is one Active Inference loop over a Vector Symbolic
> substrate, with no global backprop in steady state, no attention, no pretrained
> LM in the loop, and capabilities transformers structurally cannot have.

**CONFIDENTIAL — PATENT PENDING**
**(c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io**

---

## What this actually is

RAIN is the first generative AI architecture to compose:

- **A bipolar 10K-dim Vector Symbolic Architecture state** evolved by a HYMN MLP
  update rule that replaces transformer attention.
- **A multi-signal Expected Free Energy decoder** that draws candidates from seven
  parallel sources (HYMN prediction, LSM recurrent recall, VSA bigram, FEP action,
  sharded HRR knowledge base, Tsetlin clause vote, embedding-recalled crystals)
  and fuses them under a single calibrated distribution.
- **Local-rule learning only at runtime** — PCN, LSM-RLS, Tsetlin clause feedback,
  FEP rank-1. No global gradient ever fires once the bootstrap phase ends. The
  model gets smarter from the conversation rather than from periodic retraining.
- **A structural self-model + theory-of-mind + per-relation Bayesian calibration**
  baked into the state representation, not bolted on as prompts.
- **NSGA-II Pareto evolution + Bayesian champion/challenger promotion** across
  WASM, Rust, Go, and TypeScript runtimes, evolving the model in the background
  while it serves.

The same pipeline serves five capability surfaces: generation, dialogue, code,
reasoning, tool use.

## What this is not

- Not GPT-4. Not Claude. Not Gemini. Not at v0, not at v1.0.
- Not a fluency record-holder. RAIN's intelligence story is different.
- Not magic. Phase 1 still costs $200-$50,000 depending on scale. Continual phase
  is essentially free.

## What this gets right that no LLM does

1. **Continual learning by construction.** A→B→A retention ≥0.5 vs LLMs <0.3.
   Tell RAIN something at turn 50; it remembers at turn 5000.
2. **Calibrated uncertainty.** Every output carries an epistemic class
   (know / think / guess / unknown) and a per-relation ECE ≤0.05.
3. **Compositional generalization.** SCAN add-primitive 100%; 3/4/5-slot unseen
   compositions ≥85-100% (already measured in the RCK substrate).
4. **Structural theory of mind.** Alice's beliefs ≠ Bob's beliefs ≠ ground truth,
   maintained as first-class state objects.
5. **Grounded transparency.** Every factual claim cites a stored KB fact.
   Hallucination rate ≤1%.

## Source repositories (vendored under `third_party/`)

- **RCK** (Resonant Cognitive Kernel) — the substrate: VSA, KB, PCN, LSM, Tsetlin,
  FEP, EFE decoder, self-model, ToM, metacog, introspection.
- **Hyperion / HYMN / FERN** — the sequence engine: bipolar 10K-dim primitives,
  HYMN MLP update rule.
- **SOFAR** — frequency-banded SVD beam-steering (transplanted from transformer
  residual streams onto the VSA state).
- **Cognitive Kernel Polyglot** — operational substrate: WASM/Rust/Go/TS dispatch,
  Bayesian calibration, NSGA-II Pareto evolution, crystal schema, workload observer.
- **Evolve** — champion/challenger Bayesian beta-binomial promotion gate.

Not included: aiproof (optional input-side linter, decoupled), Rhizome
(federated VSA transport, deferred to v2).

## Status

Pre-implementation. Design approved 2026-05-22 via brainstorm. Scaffold in
progress. Implementation plan via `superpowers:writing-plans`. Target v0 tag:
12 weeks.

## Documents

- `docs/plans/2026-05-22-rain-design.md` — master design doc
- `docs/plans/2026-05-22-rain-funding-waypoints.md` — funding plan
- `docs/architecture/fep-unified-objective.md` — math derivation
- `docs/architecture/component-map.md` — module sourcing
- `docs/architecture/benchmark-suite.md` — Tier 1/2/3 spec
- `docs/design/sofar-on-vsa-transplant.md` — novel routing math
- `docs/design/multi-signal-efe-decoder.md` — seven candidate sources

## Honest framing

If you came here expecting a ChatGPT replacement at v0: this is not that.
v0 is the credible prototype of a new model class on a defensible scaling
path. v1.0 (~$20-50k train) is researcher-credible. v2.0 (federated) is
consumer-product-credible. The architecture is structured so each phase is
self-funding for the next.

If you came here expecting an LLM with extra prompting tricks: also not that.
RAIN does not use a pretrained LM at inference. The whole stack is from
first principles.

If you came here looking for the path to an AI that learns from you forever,
never forgets, tells you when it doesn't know, shows its work, models other
minds explicitly, and ships on a workstation: read on.

## License

Proprietary. SPDX NOASSERTION. See `LICENSE` and `PATENT_NOTES.md`.
