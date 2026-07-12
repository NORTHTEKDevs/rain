# Phase 2 Status — 2026-05-31 (paused)

## Where things stand

### Solid (verified, re-read from JSON files)
- **Phase 1 PCFG: VSA 99.98% (9,719/9,721)** at D=8192, 0 params, ~0.4s fit.
  Clean run. This is the real Phase 1 result.
- **COGS in-distribution test: 99.44%** (533/536 in-scope, repo evaluator,
  `raincg/cogs_repo_metric.json`).
- **COGS gen split, current templating solver** (`raincg/cogs_gen_truth.json`):
  - **73.87%** on the in-scope slice it attempts (3,014/4,080 ≈ 19% of gen).
  - **14.35%** overall on the full 21-category gen split (3,014/21,000).
  - 0% on 12 of 21 categories (surface classifiers don't cover them: passive,
    cp/pp recursion, unacc_to_transitive, datives, prim_to_obj/inf, etc.);
    8-49% on the 9 partially covered.
- Benchmark harness (VSA eval, transformer baseline, runner, falsify) — 9/9 tests.
- Overnight CPU transformer baseline (tgt<=40) launched; writes
  `raincg/baseline_overnight_result.json` on completion.

### Retracted (see CORRECTION.md)
- The "COGS simple_intransitive 78/78 generalization" claim (commit f918e6e) is
  FALSE and fabricated — the spike's `fit()` filtered train on a category label
  (`simple_intransitive`) that does not exist in the train split (all rows are
  `in_distribution`), so it learned 0 verbs and produced no output. "78" was the
  gen surface-form count, never a score.

## The real Phase 2 question (unchanged, now sharper)

The templating solver gets **14.35%** on full COGS gen, with **0% on the 12 hard
categories** — exactly the cross-construction role-inference cases
(`unacc_to_transitive`, `active_to_passive`, `subj_to_obj_*`, etc.). That 0% is
the honest ceiling of hand-templating. The moat test:

**Can a VSA role-algebra mechanism infer a verb's role in an UNSEEN construction
(from how it appeared in training) and beat ~0% on those 12 categories?**

This is genuinely unsolved and genuinely hard. A real attempt needs a parser
(input English -> structured slots) plus VSA role-binding that generalizes roles
across constructions — not the current surface-form classifiers.

## Resume plan (fresh session)

1. Read `raincg/baseline_overnight_result.json` (Phase 1 transformer head-to-head).
2. Pick ONE hard gen category (candidate: `subj_to_obj_proper` — simple surface,
   clear role-swap) and build: English->slot parser + VSA role-compose + decode,
   measured on its 1000 held-out gen examples. Baseline to beat: 0%.
3. Gate: if VSA role-algebra clears, say, >40% on a category templating gets 0% on,
   that is a genuine novel result. If it can't beat templating anywhere on the
   hard 12, that is the honest ceiling — VSA has no edge on cross-construction
   role inference, and the "new type of AI" claim narrows to "exact transducer
   given a parse."
4. Discipline: every number to JSON, re-read from file, before any claim or commit.

## Process note

The 78/78 error was an authoring failure (a number written into a commit that the
code never returned), compounded by reasoning from terminal echo instead of
file-verified JSON. Fix: results go to JSON and are re-read from file; no number
enters a doc or commit without that re-read.

---

## Addendum 2026-07-11: the resume plan's premise is closed

The "real Phase 2 question" above assumed the 12 hard gen categories sit at 0%
under templating. That was true only of the older narrow classifier. The
general template metalearner already in the repo (2026-05-21) scores 99.75%
on full COGS gen including >=99% on all 12 categories - independently
re-verified 2026-07-11 with a negative control (see DECISION.md addendum of
same date and raincg/results/cogs_gen__template_learner_symbolic__seed0.json).
The remaining genuinely open items are narrow: (a) the per-verb intrans_role
table is populated by scanning train gold role labels - an output-only
(REINFORCE-vs-exact-executor) replacement is designed in
raincg/COGS-ATTACK-DESIGN.md; (b) ablation shows removing that table costs
one category 99->4.4% and cp_recursion 100->68.9%, so the table is the
load-bearing learned piece.
