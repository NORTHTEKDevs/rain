# RAIN v1.5 -- Architecture Roadmap

> Where RAIN goes now that HYMN-Plus works. Updated 2026-05-23, day 2 of
> Phase-2 research after the HYMN-Plus breakthrough.

## What changed

Pre-2026-05-23: RAIN's design thesis was "KB-grounded continual-learning
agent" with HYMN as a placeholder sequence engine. HYMN was an MLP with
carry-step training -- it worked on Tiny Shakespeare but capped at
~1.47 NLL and would never reach LLM-class fluency.

Post-2026-05-23: HYMN-Plus is a real non-Transformer generative LM. The
RAIN architecture now consists of:

1. **Bipolar codebook substrate** (RAIN-original, kept)
2. **HYMN-Plus fluency engine** (Mamba-class selective recurrence + SwiGLU + pre-norm)
3. **KB-grounded retrieval + continual learning** (LSM/FEP/Tsetlin, RAIN-original)
4. **Cognitive surfaces exposed on every Answer** (RAIN-original, kept)
5. **Calibration tally + RLAIF judge loop** (RAIN-original, kept)

The non-Transformer fluency engine was the missing piece. We have it.

## v1.5 milestones (in order)

### M1: Validated HYMN-Plus at scale (in flight)

`hymn_plus_wt103_30k` trains on WT-103 (543 MB, dwarfs the 14M-param
model so memorization is impossible). When it finishes:

- Held-out NLL on a WT-103 tail is the headline architectural number
- Compare to old HYMN on WT-103 (2.04 NLL) -- expect HYMN-Plus to beat by 15%+
- If not, we re-tune; if yes, we have the architectural baseline

### M2: Tied-vs-untied embedding ablation

`HymnPlus.tie_weights` defaults to True (saves params, ~2x). Empirical
test: does untied output head give a measurable NLL win? Run on Tiny
Shakespeare Goldilocks recipe, both modes, 8K steps, compare held-out NLL.

### M3: BPE tokenization for HYMN-Plus

Char-level is the simplest tokenization but caps fluency. BPE (the
existing `rain.tokenize.bpe`) gives the model ~4x more semantic per
token. Plan:

- Train BPE tokenizer on the target corpus (~5 min)
- Make HymnPlus vocab_size and embed dimensions BPE-vocab-sized
- Re-run Goldilocks + scale recipes on BPE-encoded corpora
- A/B vs char-level on per-sentence-NLL (lower is better since each
  BPE token covers more chars)

### M4: HYMN-Plus on hybrid conversational corpus

Train HYMN-Plus on `data/corpora/hybrid_conversational_v1.txt`
(51K Alpaca + 7K KB-QA + 7K Shakespeare = 20.8 MB). With BPE, this
should give a v0 conversational HYMN that knows Q/A structure AND
generates semantically meaningful answers (vs the prior char-level
HYMN that only learned surface form).

### M5: Wire HYMN-Plus as the default agent fluency engine

Currently HYMN-Plus is a flag on `rain_chat --arch hymn_plus`. Make
it the default for new ConsciousAgent instances when a HYMN-Plus
checkpoint exists at the default path.

### M6: Scale-up experiments

When M1-M4 land and look healthy:

- dim=512, 8 layers, ~30M params on WT-103 (overnight)
- dim=768, 12 layers, ~90M params on WT-103 + Alpaca + WebText sample
  (this needs the SCAN-kernel work to be feasible on workstation; until
  then, rent a single H100 for $1.50/hr for 8 hours = $12)

### M7: Parallel scan kernel

The recurrence in `SelectiveGatedRecurrence.forward` is a Python loop
over T time steps. For T=64 with dim=384 this is ~150ms/step on CPU.
A proper parallel scan (Blelloch / Brent-Kung style) would be O(log T)
in operation depth instead of O(T). Implementations:

- Pure Python (numba @jit) -- 2-5x speedup
- Triton kernel (when we have CUDA) -- 10-50x speedup
- Custom Rust kernel via rain-rs -- portable, GPU-optional

M7 unblocks workstation training at dim=512+ scales.

### M8: HYMN-Plus on Hugging Face streaming dataset

Implement HF datasets streaming so we can train on TBs of data (C4,
WebText, RedPajama) without disk caching. Single-flag swap on the
training script.

### M9: Production-grade conversational agent demo

Combine:
- HYMN-Plus BPE-trained on Alpaca + KB-QA + WT-103
- Full KB seeded from Ollama (50K facts target)
- Continual learning surfaces wired
- Web UI deployed publicly

Demo flow:
1. Ask a fact-grounded question -> KB hit, citation, epistemic="know"
2. Ask a creative question -> HYMN-Plus generates, epistemic="guess"
3. Tell a new fact -> KB updates in 8ms
4. Re-ask the new fact -> KB hit
5. Ask something where KB is wrong -> human/judge feedback -> calibration updates

### M10: Public release of HYMN-Plus architecture

Once M1-M4 land:

- Open-source the architecture itself (rain.core.hymn_plus is already Apache-2.0-compat structure)
- Write the paper / blog post: "HYMN-Plus: a non-Transformer LM with
  KB grounding and continual learning, built for the workstation"
- Submit to a workshop (ICLR Workshop on Tiny Language Models,
  NeurIPS GenAI track, etc.)
- This is the credibility step for "RAIN is an AI lab" -- you need
  a public artifact people can run

## Cost / time estimate for v1.5

- M1-M5: $0 compute, ~2 weeks dev (you alone or you + occasional contractor)
- M6 with cloud: $50-$200
- M7 (scan kernel): 1-2 weeks dev, $0 compute
- M8 (HF streaming): 3-5 days dev, $0 compute
- M9 (demo): 1-2 weeks dev, $20/mo hosting
- M10 (paper/release): 2-4 weeks writing

**Total: 6-10 weeks of focused work + $100-$300 compute = a real,
published, working v1.5 RAIN.**

## What v1.5 will let you say

> "RAIN is a non-Transformer language model architecture with KB
> grounding, continual learning, calibrated transparency, and CPU/iGPU
> deployability. We released the code and a 30M-param checkpoint that
> generates coherent Q/A dialogue with audit trails. Anyone can run it
> on a laptop. The architecture is small enough to run offline, big
> enough to be useful in production domains where LLMs can't go."

That is a real category statement, defensible, with shipped artifacts.

## What v1.5 will NOT be

- Not GPT-4-class fluency at 30M params (it can't be; that's a 1000x scale gap)
- Not multimodal (kept for v2)
- Not a Claude / GPT competitor on open-ended tasks
- Not a frontier-lab competitor on training scale ($)

## Decision points / kill criteria

At each M, check:

- **M1 fails:** HYMN-Plus on WT-103 doesn't beat HYMN's 2.04 -> rethink
  arch (probably need parallel scan or attention layer added)
- **M3 fails:** BPE doesn't beat char-level on per-sentence-NLL -> 
  stay char-level for the demo, revisit BPE later
- **M4 fails:** Conversational hybrid corpus doesn't produce coherent
  Q/A outputs at 30M+ params -> add larger corpus (WebText + Alpaca)
- **M6 with cloud spent >$500 without clear improvement:** pivot to
  RAIN-on-Llama (Path-3) for product velocity while research continues
  in background

## What this roadmap is NOT

It is not the "compete with Claude" roadmap. That requires Series A
funding + 5-15 ML engineers + 2-4 years. This is the "build a credible
non-Transformer architecture with KB-grounded continual-learning
positioning that runs locally and has real shipped artifacts" roadmap.

When v1.5 is done, RAIN has:
- A working non-Transformer generative LM (shipped)
- A working continually-learning KB-grounded agent (shipped)
- A working cognitive transparency layer (shipped)
- A working local-first deployment story (shipped)
- A real paper / public artifact
- A defensible architectural moat

From there, you decide: stay broke-mode and serve regulated verticals,
or raise to scale up. Either path is open.