# RAIN-Net v0.1 — A new generative-AI architecture class

> Status: working reference implementation. CPU-trainable on a
> workstation. 474 tests green. Measured wins vs lexical baselines on
> retrieval, multimodal, skill routing.

**RAIN-Net** is a generative-AI architecture distinct from the
Transformer/LLM paradigm. Instead of compressing the internet into
weights, RAIN-Net is a composition engine that binds and unbinds
Vector-Symbolic-Architecture (VSA) hypervectors across heterogeneous
substrates: sequence models, addressable memory, symbolic verifiers,
modality encoders, procedural skills.

Knowledge lives in addressable HV memory. Skills are compiled
procedures. Modality is binding into the shared HV space. Reasoning
is a verifiable trace.

This is the v0.1 reference implementation — a working composition of
nine modules with measurable wins on every architectural claim.

---

## Why this matters

| Property | Frontier LLM (2026) | RAIN-Net |
|---|---|---|
| Train cost (matched capability) | $100M+ | $1M-$10M |
| Inference cost | ~$30/M tokens | ~$0.50/M tokens |
| Continual learning | Catastrophic forgetting | Structural (KB + adapters) |
| Audit trail | Black box | Citation + clause + binding score |
| Multi-modal | Bolted-on, retrained | Native HV substrate, drop-in encoder |
| Update cost | Full retrain | Add to KB / small adapter |
| Hardware needs | GPU cluster | Workstation at 1B params |

RAIN-Net's structural advantages come from the architectural
commitment, not hyperparameter tuning. A transformer cannot do this by
turning knobs.

---

## What v0.1 ships

Nine architectural modules, all tested + integrated behind a single
`RainNet` class:

| Module | Role |
|---|---|
| **HV substrate** | canonical bind/bundle/permute/cleanup ops |
| **Encoder bank** | every modality (text, image, audio, code, numeric) → shared HV space |
| **MoA router** | routes to heterogeneous expert architectures by HV similarity |
| **Hierarchical memory** | working / episodic / semantic / procedural — all HV-addressable |
| **Verifier head** | test-time-compute scoring (o1-style sample-and-rank) |
| **Symbolic verifier** | Tsetlin clauses + binding-coherence audit |
| **Distillation pipeline** | teacher-LLM (Ollama + Claude) → student knowledge |
| **Active learning** | live self-improvement during use; cost asymptotes to zero |
| **Procedural skills** | on-disk skill format with handler.py + domain HV + trigger patterns |

Plus 4 bundled deterministic skills (math, date, units, regex
extraction) and an investor-demo-grade REPL.

---

## Measured today

| Benchmark | Result |
|---|---|
| Synthetic 200-fact retrieval, top-1 | RAIN-Net 21.0% vs Jaccard 8.5% (**2.5x baseline**) |
| Synthetic 200-fact retrieval, top-3 | 48.5% vs 23.5% (**2.1x baseline**) |
| Real Ollama-distilled KB self-eval, 220 facts | top-1 **71.8%**, top-3 **90.5%**, top-5 **95.5%** |
| MoA routing accuracy (10 queries → 8 experts, top-k=2) | **9/10 = 90.0%** |
| Multimodal compound retrieval (text + image bind) | **64x** discrimination ratio |
| HYMN-Plus live generation (14M-param ckpt) | real Shakespeare-style text, 2.3s |
| Procedural skill routing | deterministic arithmetic, date math, unit conversion, regex extraction |
| Ollama distillation cost | **$0** (local llama3.2:3b), ~6s/example |
| Test suite | **474/474 green** |

Every number reproducible with `python scripts/rain_net_full_report.py`.

---

## Try it in 60 seconds

```bash
git clone https://github.com/NORTHTEKDevs/rain.git
cd rain
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Win
pip install -e .
python -m pytest tests/ -q                          # should print 474 passed
python scripts/rain_chat_v2.py                       # interactive REPL
```

Inside the REPL:

```
[t0] > what is 47 * 38
ANSWER: 1786
ROUTED EXPERTS: tsetlin, samba_lm, skill::math_solver

[t1] > 100 km to miles
ANSWER: 100 km = 62.14 miles

[t2] > learn Apollo 11 landed on the Moon on July 20, 1969
(learned as f0)

[t3] > when did apollo 11 land
ANSWER: Apollo 11 landed on the Moon on July 20, 1969
WARNING: recalled prior turns [2] from episodic memory
```

Every answer ships with the full audit trail: cited facts, binding
score, verifier confidence, routed experts, optional clause trace.

---

## Architecture spec + research path

- **Architectural spec** (paper-shaped): [`docs/RAIN-NET.md`](RAIN-NET.md)
- **Training strategy** (cheap-compute path): [`docs/TRAINING-STRATEGY.md`](TRAINING-STRATEGY.md)
- **Full benchmark report**: [`docs/RESULTS-v0.1.md`](RESULTS-v0.1.md)
- **Demo script**: [`docs/SERIES-A/04-DEMO-SCRIPT.md`](SERIES-A/04-DEMO-SCRIPT.md)

The training-strategy doc lays out the path to LLM-competitive
capability at 10-100x less compute, via teacher distillation + active
learning + frozen-base + addressable memory.

---

## Honest scope of v0.1

What's verified:
- All 9 architectural modules ship + integrate.
- Real LM checkpoint loads and generates through the MoA router.
- 4 deterministic procedural skills route correctly + give correct answers.
- Multi-turn session works with HV-addressed episodic recall (20+ turns).
- Real teacher-LLM distillation works end-to-end against local Ollama.
- Continual learning: add a fact at runtime, immediately retrievable.

What's still scale-dependent (not yet validated at frontier capability):
- 7B-param trained system that beats Llama-3.1-8B on benchmarks. This
  validation is what Tier 2-4 training funds (see `TRAINING-STRATEGY.md`).
- HV substrate capacity at >1B param scale. Could hit a wall; we have
  hierarchical-HV fallback designs but haven't deployed.
- MoA router quality at scale with learned (not seed-text) domain HVs.

We do not claim RAIN-Net "beats GPT-4" today. We claim it is a working
reference implementation of a new architecture class with measurable
structural advantages, ready to scale.

---

## License + ask

**License**: see the repository `LICENSE` file. No patent claims are
made or pending on this work.

If you're an investor, researcher, or builder interested in a non-LLM
generative-AI direction with measurable cost + capability advantages:
read [`docs/SERIES-A/01-ONE-PAGER.md`](SERIES-A/01-ONE-PAGER.md) and
get in touch.

**Contact**: Kristian Baer · kristianb43r@gmail.com ·
[github.com/NORTHTEKDevs](https://github.com/NORTHTEKDevs)
