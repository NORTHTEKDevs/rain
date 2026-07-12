# RAIN-CG

Vector Symbolic Architecture (VSA) compositional transduction benchmarks and
generators. This package wraps and measures the verified `pure_vsa` kernel and
hosts the Phase 2 VSA-native generator work.

## Current status (2026-07-11)

Environment fixed (torch reinstalled after venv rebuild to Python 3.14,
torch 2.13.0+cpu on py3.14.2). Fresh re-run: `pytest raincg/tests -q`
51/51 + `tests/test_scan_hyperion.py` 3/3 tests green. Overnight vanilla,
non-augmented, non-equivariant from-scratch transformer baseline confirmed
dead-before-epoch-1 (published transformer successes on this task class rely
on equivariance/augmentation/LLM-prompting, none of which this baseline has);
being rerun with checkpointing. The unified benchmark suite +
`RESULTS.md` head-to-head scoreboard exist and are populated (see below).
COGS gen headline: the general template metalearner scores 99.75% full-gen
exact match, see `DECISION.md` 2026-07-11 addendum. See `DECISION.md` for the
authoritative 2026-05-31 verdict this repo's docs are being brought into line
with.

## Phase 1 — PCFG SET head-to-head (complete)

**Claim:** zero-parameter, zero-gradient VSA algebra
(bind / unbind / bundle / permute + cleanup) solves the PCFG SET nested test by
exact algebraic evaluation, at accuracy a trained seq2seq Transformer does not
reach on the hard compositional splits — at a fraction of the compute.

### Result (verified on this machine 2026-05-31, D=8192)

| System | Params | Fit | Eval | Accuracy (exact-match) |
|---|---|---|---|---|
| VSA (`pure_vsa`) | 0 | 0.4s (CPU) | 83.5s (CPU) | **99.98%** (9,719 / 9,721) |
| Transformer (Hupkes et al. 2020, published) | ~M-scale, GPU-trained | GPU-hours | — | ~0.85 base task; **~0.50 productivity** (length) |

The Transformer row cites the original PCFG SET paper (Hupkes, Dankers, Mul,
Bruni, *Compositionality decomposed: how do neural networks generalise?*, JAIR
2020). Productivity (generalising to longer sequences) is the split where the
gap is structural: VSA's permute-composition is length-exact, so it does not
degrade on longer outputs the way a learned Transformer does.

A from-scratch Transformer baseline harness is included
(`bench/transformer_baseline.py`, `bench/run_benchmark.py --all`) for anyone who
wants to train and compare locally. It is intentionally NOT the headline number:
a small CPU-trained baseline is a weak strawman, so the published figure is cited
instead.

### Reproduce

```bash
# verified VSA result on the full 9,721-example nested test (~90s CPU)
python -m raincg.bench.pcfg_vsa_eval --d 8192
# expected: 9719/9721 = 99.98% | D=8192 | params=0

# falsification sanity: score VSA against reversed ground truth (expect ~0%)
python -m raincg.bench.run_benchmark --falsify

# optional: train+eval a local Transformer baseline (length-bounded, CPU)
python -m raincg.bench.run_benchmark --all --max-tgt-len 40 --epochs 25
```

### What this is and is not

- **Is:** evidence that the form of compositional generalisation PCFG SET tests
  is solvable by exact VSA algebra without gradient descent, deterministically,
  on CPU, with a self-falsifying exact-match metric.
- **Is not:** a general language model. It requires a per-operation handler set
  (the 10 PCFG string-edit ops), operates on a closed vocabulary, and loses to
  LLMs on any open-domain / free-text task. The input is parsed into an operation
  tree upstream; VSA does the compositional evaluation, not the parsing.

## Phase 2 — VSA-native structured-output generator (ANSWERED, 2026-07-11)

The decisive research question: can VSA algebra *generate* structured logical
forms (e.g. COGS semantic-parse outputs) by hypervector composition, replacing
hand-written per-construction templates? **Answer: solved (99.75% full-gen
exact match, `raincg/DECISION.md` 2026-07-11 addendum), but by non-VSA
symbolic/procedural composition (`pure_vsa/cogs_template_learner.py`), not by
hypervector bind/bundle/permute algebra** — consistent with the NO-GO verdict
that VSA-the-substrate is not the source of the generalization. See
`docs/plans/2026-05-31-raincg-decisive-build-design.md` for the original
framing and `raincg/DECISION.md` for the corrected reading.

### Reproduce (per RESULTS.md row -> suite stage)

Each `RESULTS.md` row is produced by one suite stage
(`python -m raincg.bench.suite --stages <name> --force`); the two COGS
template-learner stages (`cogs_template_symbolic`, `cogs_role_reinforce`)
are registered in `raincg/bench/suite.py` like the rest:

| RESULTS.md row | Suite stage / command |
|---|---|
| `pcfg_set / pure_vsa` | `--stages pcfg_vsa` |
| `pcfg_set / transformer (base, productivity)` | cited-not-reproduced, no stage |
| `scan_addprim_jump / hybrid_tagger_supervised` | `--stages scan_hybrid_supervised` |
| `scan_addprim_jump / hybrid_outputonly_reinforce` | `--stages scan_hybrid_outputonly` |
| `scan_addprim_jump / transformer_d128x3` | `--stages scan_transformer_baseline` |
| `scan_addprim_jump / grammar_induction_reinforce` | `--stages grammar_induction` |
| `scan_addprim_jump / llm_llama3.2:3b` | `--stages scan_llm_llama` (LLM `--limit 100`) |
| `scan_addprim_jump / llm_qwen2.5:14b` | `--stages scan_llm_qwen` (LLM `--limit 50`) |
| `scan_addprim_jump / NeSS, LANE, seq2seq_lstm` | cited-not-reproduced, no stage |
| `cogs_gen / llm_llama3.2:3b` | `--stages cogs_llm_llama` (LLM `--limit 100`) |
| `cogs_gen / template_learner_symbolic` (new) | `python -m raincg.bench.suite --stages cogs_template_symbolic --force` |
| `cogs_gen / template_learner_reinforce_role` (new) | `python -m raincg.bench.suite --stages cogs_role_reinforce --force` |
