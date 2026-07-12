# RAIN-Net — project conclusion (2026-05-27)

## What this project was

An attempt to build a novel non-Transformer generative-AI architecture
that beats LLMs at less compute. Across many sessions it produced
RAIN-Net v0.1 and v0.2: a composable hypervector-substrate system with
nine modules, real distillation, a multi-agent swarm, a causal-graph
reasoning layer, a reflexion loop, deterministic skills, an HTTP server,
Docker packaging, and 539 passing tests.

## What it honestly achieved

**Real and verified:**
- A clean, tested (539/539) software composition demonstrating the
  hypervector-substrate idea end-to-end.
- Retrieval parity with a production sentence-transformer baseline at
  1852-fact scale (85.8% vs 85.4% top-1) — *using the sentence-
  transformer as the encoder*.
- Genuinely unusual components bundled together: VSA substrate +
  heterogeneous experts + causal do-calculus + reflexion + HV-routed
  swarm + auditable reasoning trace.
- A real, $0-cost distillation pipeline (1632 facts, 16 domains, local
  Ollama).

**Honestly NOT achieved:**
- It did not beat LLMs. The original "novel architecture beats LLMs at
  less compute" goal is not supported by anything we measured.
- The retrieval quality came from an off-the-shelf encoder, not from
  the HV substrate.
- The generative core (HYMN-Plus, 14M params, Shakespeare-trained) is a
  toy, not a competitive language model.
- Nothing was validated at scale, peer-reviewed, or benchmarked against
  MMLU or any standard general-capability suite.

## Why it concludes here

The honest benchmark showed that as a RAG-plus-skills system, RAIN-Net
has no defensible moat — the part doing the semantic work is a model we
didn't build. Continuing to polish it would produce a competent system
with no architectural advantage. That is not the project the user set
out to build.

## Where it goes next — see PIVOT.md

The evidence points to one real path to "an architecture that beats the
dominant paradigm": **compositional generalization**, the regime where
Transformers structurally fail and where the sibling Hyperion project
already demonstrates **100% on SCAN at zero trained parameters**. The
pivot (`docs/PIVOT.md`) scopes fusing Hyperion's verified VSA
compositional kernel with RAIN-Net's reusable system layer (audit,
memory, skills) and building the head-to-head benchmark that makes the
Transformer-vs-VSA gap undeniable.

That is a narrower, honest, defensible version of the original ambition.
It is not "redefine all of AI" — but it is real, and the seed already
exists and is verified.

## Salvage list (what carries forward)

Keep: `hv_substrate.py`, `hierarchical_memory.py`, `symbolic_verifier.py`,
`causal_graph.py`, `procedural_adapter.py` + `skills/`, `reflexion.py`.

Retire the framing: "matches production RAG", "beats LLMs on cost", the
sentence-transformer-as-our-innovation retrieval claims.

Honest about: HYMN-Plus/Samba train but are toys; the wedge is the VSA
compositional kernel, not the base LM.

## Status

- Repo: github.com/NORTHTEKDevs/rain
- Tags: rain-net-v0.1, rain-net-v0.2
- Tests: 539/539 green
- License: Apache-2.0 (see LICENSE file)

Project status: **concluded as RAIN-Net; continues as the RAIN-CG pivot
per docs/PIVOT.md.**
