# RAIN Benchmark Suite — Tier 1 / 2 / 3

**Companion to:** `docs/plans/2026-05-22-rain-design.md` (§ 5)
**Status:** v0 acceptance contract. PRs claiming Tier-1 surface completion
re-run the relevant benchmark; PRs touching Tier-3-relevant code re-run
all of Tier 3.

---

## Reproducibility commitment

Every benchmark in this suite has:
- An entry-point script under `evals/`.
- A fixed random seed.
- A documented compute budget.
- A fixed corpus or held-out set (committed under `evals/data/` or referenced
  via SHA-256 + canonical URL).
- A pass threshold and a kill criterion.

One command produces all results: `python evals/reproduce.py --tier all`.

---

## Tier 1 — Five novelty-surface benchmarks

### N1 — Continual learning retention

| Property | Value |
|---|---|
| Pass threshold | `A→B→A` retention ≥ 0.50 |
| Frontier LLM today | < 0.30 (CSUR 2025 survey) |
| Corpus | Two disjoint 5K-fact task pairs (A = animals + habitats; B = chemistry + composition). |
| Procedure | Train on A → measure recall on A → continual-train on B → measure recall on A again. Retention = `recall_after_B / recall_initial`. |
| Kill criterion | < 0.40 |
| Entry point | `evals/tier1_novelty/retention.py` |

### N2 — Calibrated uncertainty

| Property | Value |
|---|---|
| Pass threshold | ECE ≤ 0.05 across 14 relation types, 1000-question probe |
| Frontier LLM today | 0.15-0.25 (Lin et al. 2022) |
| Corpus | 1000 questions stratified across 14 RCK relation types, mix of known + held-out facts. |
| Procedure | Bucket predicted-confidence into 10 bins; ECE = Σ |bin_accuracy - bin_confidence| × bin_weight. |
| Kill criterion | ECE > 0.10 |
| Entry point | `evals/tier1_novelty/calibration.py` |

### N3 — Compositional generalization (SCAN)

| Property | Value |
|---|---|
| Pass threshold | 100% on add-primitive; ≥ 95% on 3-slot, ≥ 90% on 4-slot, ≥ 85% on 5-slot unseen combos |
| Frontier LLM today | GPT-4 ~70% on add-primitive; drops fast with slot count |
| Corpus | SCAN add-primitive split + RCK v1.1 3/4/5-slot composition probe |
| Procedure | Hold out 90% of combinations, train on 10%, evaluate on held-out 90%. |
| Kill criterion | < 100% on add-primitive |
| Entry point | `evals/tier1_novelty/scan.py` (vendored from `hyperion/eval_scan.py`) |

### N4 — Structural theory of mind

| Property | Value |
|---|---|
| Pass threshold | 10/10 Sally-Anne; 18/20 BDI tracking over 20-turn story |
| Frontier LLM today | LLMs fail Sally-Anne variants ~40% (Ullman 2023) |
| Corpus | Sally-Anne canonical + 10 variant tests; 5-story BDI dialogue suite |
| Procedure | First-order belief queries on canonical; second-order on variants; per-turn belief-state checks on dialogue. |
| Kill criterion | < 8/10 Sally-Anne |
| Entry point | `evals/tier1_novelty/tom.py` |

### N5 — Grounded transparency

| Property | Value |
|---|---|
| Pass threshold | ≥ 98% factual-claim citation coverage; hallucination rate ≤ 1% |
| Frontier LLM today | RAG-augmented LLMs ~30-50% citation coverage; 70-90% hallucination on novel facts |
| Corpus | 100-claim audit set across 5 domains |
| Procedure | Each output's factual claims auto-extracted; each claim checked for a citation chain back to a KB write. Hallucination = factual claim with no traceable KB origin. |
| Kill criterion | < 95% citation coverage or > 5% hallucination |
| Entry point | `evals/tier1_novelty/transparency.py` |

---

## Tier 2 — Eight LLM-parity-at-toy-scale benchmarks

### L1 — Tiny Shakespeare val-loss

| Pass threshold | ≤ 1.55 nats/char |
| Baseline | nanoGPT-char at matched FLOPs |
| Corpus | Andrej Karpathy's Tiny Shakespeare (1.1MB) |
| Entry point | `evals/tier2_llm_parity/tiny_shakespeare.py` |

### L2 — WikiText-2 perplexity (small subset)

| Pass threshold | ≤ 80 perplexity |
| Baseline | nanoGPT-medium ~80 |
| Corpus | WikiText-2 first 100K tokens |
| Entry point | `evals/tier2_llm_parity/wikitext.py` |

### L3 — HumanEval-tiny

| Pass threshold | ≥ 40% pass@1 |
| Baseline | GPT-2 ~10% |
| Corpus | 10 hand-picked easy HumanEval problems |
| Entry point | `evals/tier2_llm_parity/humaneval_tiny.py` |

### L4 — GSM8K-tiny

| Pass threshold | ≥ 40% |
| Baseline | Pythia-160M ~15% |
| Corpus | 5 grade-school math problems |
| Entry point | `evals/tier2_llm_parity/gsm8k_tiny.py` |

### L5 — MMLU-tiny

| Pass threshold | ≥ 30% |
| Baseline | Random 25%; Pythia-410M ~28% |
| Corpus | 3 MMLU categories × 50 questions each |
| Entry point | `evals/tier2_llm_parity/mmlu_tiny.py` |

### L6 — MTBench-tiny

| Pass threshold | ≥ 3.0 / 10 (judge model: external eval pipeline, not RAIN) |
| Baseline | nanoGPT-medium ~2.0 |
| Corpus | 10 MTBench dialogue prompts |
| Entry point | `evals/tier2_llm_parity/mtbench_tiny.py` |

### L7 — Tool-use success

| Pass threshold | ≥ 8/10 |
| Baseline | GPT-3.5 ~9/10 on simple tasks |
| Corpus | 10 function-call tasks (calculator, search, file-read, etc.) |
| Entry point | `evals/tier2_llm_parity/tool_use.py` |

### L8 — Inference latency

| Pass threshold | ≤ 10 ms / token on CPU at toy scale |
| Baseline | RCK v1.0 measured 5.5 ms/token |
| Hardware | Standard workstation single-thread CPU |
| Entry point | `evals/tier2_llm_parity/latency.py` |

---

## Tier 3 — Five architectural-soundness checks

### A1 — EFE source-mix sanity

Each of the 7 candidate sources contributes ≥ 5% to the final token choice
on a balanced probe set (100 prompts spanning generation / dialogue / code /
reasoning / tool-use).

Entry: `evals/tier3_soundness/efe_source_mix.py`

### A2 — SOFAR-on-VSA improvement

Val-loss with SOFAR routing < val-loss without, by ≥ 3%. Ablation: same
model with and without `rain/routing/` enabled.

Entry: `evals/tier3_soundness/sofar_ablation.py`

### A3 — NSGA-II promotion rate

≥ 1 champion-displacing variant promoted per 100 generations over a 1000-gen
background run.

Entry: `evals/tier3_soundness/nsga2_promotion.py`

### A4 — Local-rule continual stability

No NaN, ECE drift ≤ 0.1, no KB corruption over a 10K-turn continual run on
the RAIN benchmark dialogue set.

Entry: `evals/tier3_soundness/continual_stability.py`

### A5 — Polyglot dispatch correctness

WASM, native Rust, native Go, and TypeScript runtimes produce bit-identical
outputs on identical input (modulo float ε = 1e-9).

Entry: `evals/tier3_soundness/dispatch_correctness.py`

---

## Comparative bar against non-LLM alternatives

The v0 model must match or beat the following on Tier 2 at matched FLOPs while
having all Tier 1 capabilities (which the comparators don't):

| Comparator | Approximate scale | Status |
|---|---|---|
| RWKV-7-mini | ~150M params | Public weights |
| LFM2-small | ~350M | Public weights (Liquid AI 2025) |
| Mamba-3-small | ~150M | Public weights |
| Pythia-160M | 160M | Public weights |

All four are run inside `evals/baselines/` under matched-FLOPs settings.

---

## Running the full sweep

```
python evals/reproduce.py --tier 1
python evals/reproduce.py --tier 2
python evals/reproduce.py --tier 3
python evals/reproduce.py --tier all  # ~60 minutes at toy scale
```

Results are written to `evals/results/<date>/<tier>/<benchmark>.json` with
per-run seeds, hardware fingerprint, and pass/fail decision.

---

## v0 acceptance contract

The repository is tagged `v0.1.0` only when **all 18 benchmarks pass** their
thresholds on a single reproducible run on a workstation 4090 + CPU.

Any failure must either:
1. Trigger the kill criterion → architecture change documented in design doc,
2. Be remediated by an explicit follow-up PR before tagging, or
3. Be explicitly waived in CHANGELOG with rationale (only Tier 3 checks
   eligible for waiver, never Tier 1 or Tier 2 L1).
