# RAIN — Resonant Active Inference Network

> Honest compositional-generalization research. Exact symbolic composition +
> learned selection reaches **100% on SCAN** and **99.75% on COGS gen** at
> seconds of CPU — measured head-to-head against transformer and LLM
> baselines on the same splits, with every number backed by a checked-in
> result artifact and a reproduction command.

**License:** Apache-2.0 · **Status:** research, actively maintained ·
**Scoreboard:** [`raincg/RESULTS.md`](raincg/RESULTS.md)

---

## The headline (measured, reproducible)

| Benchmark | Exact-composition system | Same-split baselines (measured here, direct few-shot) |
|---|---|---|
| SCAN `addprim_jump` (7,706 held-out) | **100%** — 65-param output-only REINFORCE tagger + exact executor; also 100% supervised | gpt-5.5: **86%** · claude-sonnet-5: **72%** · qwen2.5:14b: **12%** · llama3.2:3b: **0%** · from-scratch transformer (609k params): **0%** |
| COGS gen (21,000) | **99.75%** — symbolic template metalearner fit on the train split only (~4s CPU, zero gradient descent); a 122-param output-only REINFORCE variant matches it | gpt-5.5: **46%** · claude-sonnet-5: **36%** · llama3.2:3b: **0%** |
| PCFG SET (paired, tgt≤40, n=1000) | **100%** — pure VSA transduction, 0 params, 0.4s fit | transformer (15.7M params, 108 min CPU, capped budget): **0.5%** |

Full tables with 95% CIs, parameter counts, compute, API-error disclosure,
and evidence tiers (`measured-fresh` vs `cited-not-reproduced`) are
generated into [`raincg/RESULTS.md`](raincg/RESULTS.md). Published
decomposition prompting (least-to-most) reaches ~99% on these splits with
frontier LLMs — those results are cited in the tables rather than
reproduced. **The claim here is not "LLMs can't do this."** Frontier
models measured here reach 72-86% on SCAN with plain few-shot prompting.
The claim is the remaining gap and the cost of closing it: exact,
deterministic, auditable compositional generalization at seconds of CPU,
versus probabilistic answers at API inference cost.

## What the honest mechanism is (and is not)

Three findings, each with its artifact:

1. **The generalization credit belongs to exact, content-independent
   composition — not to hypervectors.** An adversarial "moat test"
   ([`raincg/MOAT-VERDICT.md`](raincg/MOAT-VERDICT.md),
   [`raincg/DECISION.md`](raincg/DECISION.md)) showed the COGS
   cross-construction win comes from construction-keyed role binding in
   plain code; the VSA layer is an exact, lossless transducer *given*
   structure. We keep that verdict published because it is true.
2. **Learned selection + exact operations is the pattern that works.**
   Gradient (or REINFORCE from output-only reward) learns *which* roles,
   lexicon entries, and templates apply; algebraic composition executes
   them exactly. That hybrid hits the numbers above where pure gradient
   models memorize and collapse.
3. **The caveat that stays attached:** the COGS template metalearner
   encodes the task's role ontology and construction signatures — a
   task-specific inductive bias. This is not a general language learner,
   and we do not claim it is.

## Reproduce everything

```bash
git clone https://github.com/NORTHTEKDevs/rain.git && cd rain
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[raincg,dev]"   # raincg = torch for the benchmark; dev = pytest

python scripts/download_data.py        # SCAN + COGS + PCFG from their public upstreams (row-count verified)

python -m pytest raincg/tests/ -q      # benchmark harness + headline regression floors
python -m pytest tests/ -q             # RAIN-Net system suite (539 tests)

python -m raincg.bench.suite           # list all benchmark stages
python -m raincg.bench.suite --stages cogs_template_symbolic --force   # COGS 99.75% in ~3s
python -m raincg.bench.suite --stages scan_hybrid_outputonly --force   # SCAN 100% in ~2.5min
python -m raincg.bench.results_table --out raincg/RESULTS.md           # regenerate the scoreboard
```

Every row in `RESULTS.md` maps to one suite stage
([`raincg/README.md`](raincg/README.md) has the full row→command table).
LLM baseline stages need a local [Ollama](https://ollama.com) or an
OpenRouter API key (`OPENROUTER_API_KEY`); the long transformer trainings
are checkpointed and resumable.

## Why this repo reads differently from most research repos

This project retracted its own headline result once
([`docs/CONCLUSION.md`](docs/CONCLUSION.md)): an early "architectural moat
validated" claim turned out to be measured on training data, and a later
fabricated benchmark number was caught and documented
([`raincg/CORRECTION.md`](raincg/CORRECTION.md)). The discipline that came
out of that is the repo's actual contribution style:

- **No number without its artifact.** Every claim traces to a JSON in
  [`raincg/results/`](raincg/results/) produced by a checked-in script.
- **Negative controls everywhere.** Shuffled-target and label-flip controls
  ship with the headline results; evals that can't fail are treated as
  broken.
- **Steelman baselines, cited.** Where a published technique beats or
  matches us (least-to-most prompting, LeAR, NeSS/LANE), it is in the
  table, labeled `cited-not-reproduced`.
- **Corrections are appended, never erased.** Superseded claims carry dated
  addenda pointing at the fresh measurement.

## Project history (kept, not erased)

The full RAIN-Net-era README — including the v5 retraction tables and the
2026-05-24 honest-status audit — is preserved verbatim at
[`docs/HISTORY-RAIN-NET-README.md`](docs/HISTORY-RAIN-NET-README.md).

## What's in the repo

- `rain/` — RAIN-Net: the 9-module hypervector composition system
  (encoder bank, mixture-of-adapters router, hierarchical memory,
  verifier head, symbolic verifier, causal graph, reflexion loop,
  deterministic skills, HTTP serve layer). 539 tests.
- `raincg/` — the compositional-generalization benchmark: suite runner,
  results generator, measured artifacts, and the honesty record
  (DECISION / MOAT-VERDICT / CORRECTION).
- `pure_vsa/`, `vsa_core/` — the exact-composition kernels (bind /
  bundle / permute / cleanup, SCAN & COGS & PCFG solvers), published from
  the sibling Hyperion research line.
- `experiments/` — the head-to-head baseline scripts (transformer,
  local-LLM, frontier-LLM via OpenRouter, hybrids, grammar induction).
- `rain-rs/` — Rust hot kernels (VSA ops, Tsetlin, NSGA-II, SVD routing).
- `docs/` — architecture spec ([`RAIN-NET.md`](docs/RAIN-NET.md)), the
  pivot record ([`PIVOT.md`](docs/PIVOT.md)), and the project's honest
  conclusion ([`CONCLUSION.md`](docs/CONCLUSION.md)).

## Citing

If you use the benchmark harness or the COGS/SCAN solvers, cite this
repository. The datasets belong to their authors: SCAN (Lake & Baroni,
2018), COGS (Kim & Linzen, 2020), PCFG SET (Hupkes et al., 2020).

---

*(c) 2026 Kristian Baer / FrostByte LLC (Northtek). Apache-2.0.*
