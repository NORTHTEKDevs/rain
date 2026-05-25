# Deep Research Push — November 2025

> Goal: find an architectural direction that could dramatically improve
> RAIN. Three independent research agents (Perplexity, Codex via O3,
> Grok) investigated different angles in parallel. The honest finding
> is documented here without spin.

## TL;DR

**No 2024-2025 architecture has been shown to dramatically beat
Transformers (5-10x at matched compute).** The Bitter Lesson still
holds. Every "novel architecture" with verified peer-reviewed numbers
shows incremental gains (~0.15-0.20 perplexity points), typically at
specific context-length regimes. None is a revolution.

The single most-promising IMPLEMENTABLE direction with verified numbers
is **Samba** (Microsoft Research, ICLR 2025): Mamba + Sliding-Window
Attention interleaved. Implemented in `rain/core/hymn_samba.py` after
this research. Expected payoff at our scale: modest (~0.1-0.3 nats/char).

## Verified architecture comparison (from the literature)

### Samba (the implementable winner)

ICLR 2025, arXiv:2406.07522, code at github.com/microsoft/Samba.

| Model (438M params, Pile) | 4k ctx | 8k ctx | 16k ctx |
|---|---|---|---|
| Llama-2 | 11.14 | 47.23 | 249.03 |
| Mamba | 10.70 | 10.30 | 10.24 |
| **Samba** | **9.65** | **9.65** | **9.57** |

| Model (1.3B params, 100B tokens) | 4k | 8k | 16k |
|---|---|---|---|
| Llama-2 | 7.60 | 44.32 | 249.64 |
| Mamba | 7.47 | 7.26 | 7.15 |
| **Samba** | **7.32** | **7.11** | **6.96** |

The qualitative win: Llama-2 degrades catastrophically past training
context (4k → 16k goes from 11 PPL to 249 PPL). Mamba and Samba don't.
Samba beats pure Mamba by ~0.15-0.20 PPL at this scale.

### Other directions evaluated (rejected for our use)

| Direction | Status | Why rejected |
|---|---|---|
| xLSTM | Beats Mamba ~0.2 pts | Single-lab result, thin external repro |
| TTT (test-time training) | Wins at 8k+ context | Requires custom CUDA kernels |
| EDLM (energy-based diffusion) | 49% over best diffusion baseline | Still uses Transformer internals; not a replacement |
| Memory Layers at Scale | 128B memory params at 1T tokens | Inside a Transformer; not an alternative arch |
| **VSA/HDC generative LM** | **No competitive published result** | Field doing classification/reasoning, not generative |
| Tsetlin Machine generative | Caps at LSTM-class (HVTM Aug 2024) | Not Transformer-competitive at scale |
| Active Inference / FEP | Not a generative arch in practice | All recent papers use AIF as control layer |
| MoE under 1B params | Routing overhead eats sparse-compute benefit | Helps only at 7B+ + GPU |
| SEDD (discrete diffusion) | Matches GPT-2 1.5B | Doesn't beat modern AR; incompatible with causal streaming |

### What the contrarian research found

From the Grok contrarian analysis:

> "No novel-architecture startup has shipped a model that outperforms
> Llama/Qwen at matched parameter counts on standard benchmarks as of
> mid-2025. The user's premise — that a novel non-Transformer
> architecture can compete with or beat LLMs at dramatically less
> compute across general tasks — is contradicted by the 2022-2025
> empirical record."

The practical small-model path that's actually working: **distillation
from larger models** (Phi-3.5 Mini, Qwen 2.5 0.5B-3B, Llama 3.2 1B).
Those win at their scale because they inherit knowledge compression
from giant teachers — which structurally requires the LLM ecosystem.

### Startup reality check

- **Liquid AI** ($250M raised): No public benchmark vs Llama/Qwen on NLP. Marketing-to-paper ratio is high.
- **Sakana AI**: Evolutionary model merging + AI Scientist. Research artifacts, not arch victories.
- **AI21 (Jamba)**: Hybrid SSM-Transformer, competitive at long context. AI21 has laid off staff; commercial future uncertain.

## What we built in response

`rain/core/hymn_samba.py` — clean Samba implementation: alternating
MambaLayer (our existing S6 + SwiGLU) and SwaLayer (sliding-window
attention + SwiGLU). 11 tests pass, including all causality + window-
respect checks.

The architecture is in main. Whether to actually train it at scale
is a separate decision that depends on a compute budget.

## Where RAIN actually has opportunity (honest)

The research converges on three areas where small-team architectural
work in 2024-2025 actually pays off:

1. **Narrow domain with specialized inductive bias** — a small
   non-Transformer model with hard-coded domain structure (genomics,
   physical simulation, formal logic, regulated knowledge) can beat
   general LLMs in that domain at 10-100x less compute. Requires
   accepting the model does ONE thing well.

2. **Edge inference efficiency** — SSMs and linear attention have
   genuine advantages for streaming, low-memory, latency-critical
   deployment. A team building an edge runtime for a specific task
   class has a real engineering opportunity. Not a general-AI opportunity.

3. **Hybrid systems with retrieval + symbolic** — composing a small
   specialized model with KB grounding, structured knowledge, and
   symbolic reasoning to handle what brute-force LLMs do poorly. This
   is where RAIN-style thinking has legitimate footing — but only if
   framed as "better system for specific tasks" not "better general
   intelligence at less compute."

## The honest reframe

The user's framing "novel architecture that beats LLMs at less
compute, redefines AI" is **not supported by 2024-2025 empirical
evidence**.

RAIN's honest competitive lane: a specialized system that outperforms
LLMs on specific task classes where its architectural priors match
the problem structure (auditable, continually learning, KB-grounded
verticals like legal/medical/finance/regulated knowledge work).

This is achievable with a small team. The broader "redefine AI"
claim is a research-aesthetic preference, not an evidence-supported
commercial strategy.

## Sources

- [Samba: Simple Hybrid State Space Models (ICLR 2025)](https://arxiv.org/abs/2406.07522)
- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [TTT: Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620)
- [Mamba S6: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)
- [Striped Hyena (Mechanistic Design)](https://arxiv.org/abs/2403.17844)
- [EDLM: Energy-Based Diffusion LMs (ICLR 2025)](https://arxiv.org/abs/2410.21357)
- [Memory Layers at Scale - Meta FAIR](https://arxiv.org/abs/2412.09764)
- [LARS-VSA: VSA for Abstract Rules](https://arxiv.org/abs/2405.14436)
- [HVTM: Hyperdimensional Tsetlin Machines](https://arxiv.org/abs/2408.16620)
- [MoE Scaling Laws](https://arxiv.org/abs/2502.05172)
- [SEDD: Discrete Diffusion Modeling (ICML 2024)](https://arxiv.org/abs/2310.16834)
- [ARMT: Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841)
- [Retro-li: Small-Scale RAG (ECAI 2024)](https://arxiv.org/abs/2410.00004)
- [Latent Space: 2024 Post-Transformer Architectures Review](https://www.latent.space/p/2024-post-transformers)
