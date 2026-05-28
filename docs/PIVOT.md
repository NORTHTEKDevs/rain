# RAIN-Net → RAIN-CG: the honest pivot

> Written 2026-05-27 at project conclusion. This document records why
> the project pivots from "RAG system that matches production" to
> "compositional-reasoning core that decisively beats Transformers on a
> verifiable axis," and scopes that pivot concretely for whoever picks
> this up next (likely future-me).

## Why pivot at all

The original goal was: *a novel non-Transformer architecture that beats
LLMs at less compute and redefines AI.* Across this project we built
RAIN-Net v0.1 → v0.2: a 9-module hypervector-substrate composition with
539 passing tests, real distillation, multi-agent swarm, causal graph,
reflexion loop, and a clean demo surface.

Then we ran the honest benchmark we'd been avoiding, and it told the
truth:

- Our own n-gram HV encoder **lost** to a stock sentence-transformer by
  16 points at top-1 retrieval (62.9% vs 78.8%).
- We only reached parity (85.8% vs 85.4% at 1852-fact scale) **after we
  swapped our encoder out for the sentence-transformer itself** and
  bipolarized its output into HV space. The retrieval quality is
  *theirs*, not ours.
- The actual generative "novel AI" core (HYMN-Plus) is a 14M-param
  Shakespeare model that emits period-flavored gibberish.

Conclusion: RAIN-Net as built is **a competent RAG-plus-skills system
wrapped around an off-the-shelf encoder.** It is not a new architecture
that beats LLMs. Continuing down the "match production RAG" road leads
to "fine system, no moat."

## The reframe that is actually true

You do not displace a dominant paradigm by being broadly better. You
displace it by being **decisively better in a regime where the
incumbent has a structural, not-just-tuning weakness**, then expanding
outward. Mamba beat Transformers at long-context streaming (attention is
O(n²)); it did not beat them generally.

Transformers have published, reproducible, **structural** weaknesses
that do not close with scale:

1. **Systematic compositional generalization.** On SCAN's hard splits
   (e.g. `addprim_jump`, `length`), large Transformers score ~0-20%.
   VSA / symbolic methods can hit ~100%.
2. **Continual learning.** Catastrophic forgetting is architectural.
3. **Length / OOD generalization.** They degrade past trained sequence
   lengths.

## The asset we already have

This is the part that makes the pivot real rather than aspirational:

**The sibling project Hyperion's `pure_vsa` track gets 100% on all
seven SCAN splits (27,141 / 27,141), with ZERO trained parameters.**
(Verified; repro `pytest tests/test_scan_hyperion.py`; paper draft at
`pure_vsa/PAPER.md`.) A Transformer of any size gets a fraction of that
on the compositional splits.

That is a genuine "we beat the dominant paradigm decisively at ~zero
compute" result. It exists today. It is narrow — but it is real, and it
is exactly the kind of structural-weakness wedge that can actually
displace an incumbent in a domain.

## Scope of the pivot — RAIN-CG (Compositional Generalization)

**Thesis:** *On the class of problems where Transformers provably fail
to generalize, a VSA-native architecture succeeds at orders of magnitude
less compute. Build the architecture and the benchmark suite that make
that gap undeniable; then expand the regime outward.*

### Phase 0 — fuse the two assets (weeks 1-3)
- Take Hyperion's verified `pure_vsa` SCAN core as the reasoning kernel.
- Keep the RAIN-Net system layer that is genuinely useful and proven:
  the HV substrate (`hv_substrate.py`), hierarchical memory, symbolic
  verifier / audit trail, and the skill registry.
- Drop the parts that were RAG-wrapper theater: the "matches production"
  framing, the Series-A "beats LLMs on cost" pitch, the sentence-
  transformer-as-our-encoder retrieval claims.
- Deliverable: one repo where the compositional kernel (the part that
  beats Transformers) is the headline, and the system layer (audit,
  memory, skills) is the supporting cast.

### Phase 1 — make the gap undeniable (weeks 3-8)
- Build the head-to-head benchmark harness:
  - SCAN (all splits) — VSA core vs same-budget Transformer.
  - COGS — compositional semantic parsing.
  - Length-generalization probe — train short, test long.
  - A continual-learning protocol (A→B→A retention) — the existing
    `evals/tier1_novelty/retention.py` is a starting point.
- For each: report VSA-core score, Transformer-of-matched-params score,
  and the compute ratio. The claim is the *gap plus the compute ratio*,
  nothing more.
- Honesty gates (carry these forward from this project's hard-won
  lessons): never claim "beats LLMs" without the actual same-split
  baseline number in-session; novelty = "synthesis", never "first";
  read the eval before believing the eval.

### Phase 2 — publish + expand the regime (months 3-9)
- Write it up as what it honestly is: *"VSA-native compositional
  generalization at near-zero compute where Transformers fail."* This is
  a real, submittable workshop/conference paper. Hyperion's
  `pure_vsa/PAPER.md` is the seed.
- Then push the regime outward: which adjacent tasks share the
  compositional-structure property? Program synthesis, formal-logic QA,
  structured-data transformation, instruction-following with novel
  combinators. Each one you can win is one more brick.

### What success looks like (and does not)
- **Success:** a defensible research contribution + a startup wedge:
  "the system you use when compositional correctness matters and
  Transformers hallucinate the recombination." Auditable, continually
  learning, cheap.
- **Not success, and do not claim it:** a general assistant that beats
  ChatGPT. The path from "wins on compositional generalization" to
  "general intelligence" is long and may not exist. Do not sell it.

## What to keep from RAIN-Net (the salvage list)

Genuinely reusable, already tested:
- `rain/core/hv_substrate.py` — clean VSA primitives. Keep.
- `rain/core/hierarchical_memory.py` — addressable memory. Keep.
- `rain/core/symbolic_verifier.py` — audit trail / binding coherence. Keep.
- `rain/core/causal_graph.py` — do-calculus over triples. Keep (real,
  unusual).
- `rain/core/procedural_adapter.py` + `skills/` — deterministic skills.
  Keep.
- `rain/cognition/reflexion.py` — verifier-gated self-critique. Keep.

Demote / archive (RAG-wrapper, not the moat):
- `rain/core/encoder_bank.py` learned-encoder path — fine as a utility,
  but stop calling it our innovation.
- `rain/core/two_stage_retrieve.py` — useful infra, not a research
  contribution.
- The Series-A "beats LLMs on cost" framing — retired.

Honest about the base LM:
- HYMN-Plus / HYMN-Samba are real non-Transformer sequence models that
  *train*, but at 14M params on Shakespeare they are toys. They are not
  the wedge. The wedge is the VSA compositional kernel.

## One-paragraph version for whoever reads this next

RAIN-Net proved that a hypervector-substrate composition is buildable
and clean, but also proved (honestly, via its own benchmark) that as a
RAG system it has no moat — the retrieval quality came from an
off-the-shelf encoder. The real, evidence-backed path to "an
architecture that beats the dominant paradigm" is narrow and specific:
VSA-native **compositional generalization**, where Transformers
structurally fail and where the sibling Hyperion project already
demonstrates 100% on SCAN at zero trained parameters. Pivot the focus to
that kernel, build the head-to-head benchmark that makes the gap
undeniable, keep RAIN-Net's audit/memory/skill layer as supporting
infrastructure, and stop selling "beats LLMs generally." That is the
most real version of the original dream the evidence actually supports.
