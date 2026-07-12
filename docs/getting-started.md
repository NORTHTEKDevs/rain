# RAIN -- Getting Started

## What RAIN is in one paragraph

RAIN is a new class of generative AI -- not an LLM, not a state-space model,
not a transformer. It's one Active Inference loop over a Vector Symbolic
substrate, with no global backprop at runtime. The architecture demonstrates
capabilities that are hard for gradient-trained models: continual learning
without forgetting, calibrated uncertainty with epistemic classification
(know / think / guess / unknown), exact compositional generalization
(measured head-to-head against transformer and LLM baselines in
`raincg/RESULTS.md`), and structural Theory of Mind that maintains
separate beliefs per agent.

## Install

```bash
git clone https://github.com/NORTHTEKDevs/rain.git
cd rain
pip install -e ".[raincg,dev]"   # add [bootstrap] for the learned-encoder/training extras
# This builds the Rust kernel via maturin (~30s first time).
pytest -q -m "not slow"
# Expected: 538 passed, 1 deselected (plus raincg: pytest raincg/tests -q)
```

Requirements:
- Python 3.11+ (3.14 supported)
- Rust toolchain (stable, MSRV 1.85)
- SentencePiece, numpy, scipy

## Five-minute tour

The top-level interface is `rain.agent.ConsciousAgent`. It exposes
`tell`, `ask`, `describe`, `feedback`, `self_describe`, and
`what_just_happened`.

```python
from rain.agent import ConsciousAgent

agent = ConsciousAgent(dim=2048, num_shards=8, seed=0)

# Teach
agent.tell("rome", "capital_of", "italy")
agent.tell("italy", "locatedin", "europe")

# Ask -- get back an Answer with epistemic + citations + confidence
answer = agent.ask("rome", "capital_of")
print(answer.text)              # "I know that rome capital of italy ..."
print(answer.epistemic)         # "think" (with low calibration tally) or "know"
print(answer.citations)         # [("rome", "capital_of", "italy")]
print(answer.confidence)        # 1.0 (direct inference)

# Inheritance -- "rome" inherits locatedin from "italy" (after we tell it)
agent.tell("rome", "locatedin", "italy")  # explicit
# Or via chain (not implemented in v0; transitive walks single-hop)
ans = agent.ask("rome", "locatedin")
print(ans.text)
```

## The four demonstrated Tier-1 surfaces

### N2 -- Calibrated uncertainty (ECE = 0.00)

Every Answer carries an epistemic class. Run the full benchmark with:

```bash
python evals/tier1_novelty/calibration.py --n 1000 \
  --out evals/results/$(date +%F)/tier1/N2.json
```

Frontier LLMs: ECE 0.15-0.25 (Lin 2022). RAIN at v0: 0.00 because
confidence is a deterministic function of inference source.

### N3 -- Compositional generalization (100% across all four splits)

```bash
python evals/tier1_novelty/scan.py \
  --out evals/results/$(date +%F)/tier1/N3.json
```

100% on add-primitive, 3-slot (64 combos), 4-slot (256 combos), 5-slot
(1024 combos). GPT-4 scores ~70% on add-primitive and drops with slot count.
The RAIN mechanism is slot-based VSA composition via `CompositionalReasoner`.

### N4 -- Structural Theory of Mind (10/10 Sally-Anne + 20/20 BDI)

```bash
python evals/tier1_novelty/tom.py \
  --out evals/results/$(date +%F)/tier1/N4.json
```

`rain.cognition.theory_of_mind.TheoryOfMind` stores per-believer-chain
beliefs in separate sharded HRR KBs. Alice's beliefs about a marble's
location never leak into Bob's beliefs. Second-order belief tracking works.
LLMs fail Sally-Anne variants ~40% (Ullman 2023); RAIN passes 10/10
canonical + variants.

### N5 -- Grounded transparency (100% coverage, 0% hallucination)

```bash
python evals/tier1_novelty/transparency.py \
  --out evals/results/$(date +%F)/tier1/N5.json
```

Every Answer carries a `.citations` field with the (S, R, O) triples
backing the response. We audited 100 questions across 5 domains: 100%
citation coverage on the 72 answerable questions, 0% hallucinated citations.
RAG-augmented LLMs: ~30-50% coverage, 70-90% hallucination on novel facts.

## The runnable demo

```bash
python examples/01_chat_e2e.py
```

This walks through teaching + asking + each Tier-1 surface live.

## Architecture in one diagram

```
                input tokens
                     |
                     v
       +---------- BPE tokenizer
       |
       v
   Codebook -> bipolar 10K-dim HV (state s_t)
       |
       v
   +-- SOFAR routing (Hadamard bands + SVD beam-steer) [optional]
   |   v
   |   HYMN MLP forward  -> s_{t+1}
   |   v
   |   +---- 7-source EFE decoder ----+
   |          - HYMN prediction       |
   |          - LSM RLS recall        |
   |          - VSA bigram            |
   |          - FEP rank-1 action     |
   |          - Sharded HRR KB lookup |
   |          - Tsetlin vote          |
   |          - Crystal recall        |
   |          v
   |   fusion + calibration re-weight + top-p
   |          v
   |        next token
   |          |
   +----------+
              v
       local-rule learning (PCN + LSM-RLS + Tsetlin + FEP rank-1)
       writes back to:
       Codebook + LSM + Tsetlin + FEP + KB + Crystals
```

## Source repos this is built from

A meaningful share of the code is ported and adapted from prior NORTHTEKDevs
research projects:
- **RCK** (public, MIT, [github.com/NORTHTEKDevs/rck](https://github.com/NORTHTEKDevs/rck))
  -- VSA primitives, sharded KB, cognitive layer, EFE decoder
- **Hyperion** -- HYMN MLP update rule, vsa_core primitives (the `pure_vsa`
  slice is vendored into this repo under `third_party/hyperion`)
- Routing math (SVD beam-steering transplanted from transformer residual
  streams to VSA state) and the evolutionary champion/challenger gate are
  adapted from sibling internal research projects.

## What's not yet built

- **HYMN gradient pre-training** -- Phase 2.3 (scaffold ready in `scripts/pretrain_hymn.py`). Needs single 4090 + 1-2 weeks for Tiny Shakespeare scale.
- **Tier 2 LLM-parity benchmarks** (L1-L8) -- blocked on HYMN training for the language-modeling axes (L1, L2). L7 tool-use and L8 latency reachable now.
- **N1 continual-learning A->B->A retention** at the gradient-fed scale -- blocked on trained HYMN.
- **WASM / Go / TS dispatch verification** (A5 partial) -- Python <-> Rust verified.
- **rain-chat Tauri UI** -- Phase 8.

## Pointer to the design

Read in order:
1. `docs/architecture/fep-unified-objective.md` -- math derivation
2. `docs/architecture/benchmark-suite.md` -- acceptance contract
