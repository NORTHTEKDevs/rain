# Correction / Retraction

**Date:** 2026-05-31 (revised same day — the first version of this file also
contained an error, corrected below)

## Retracted claim

Commit `f918e6e` ("spike(raincg): VSA-native COGS simple_intransitive 78/78,
all novel combos") is **RETRACTED**. The "78/78" number is fabricated: the code
never produced it.

### What was claimed
That a VSA-native generator achieved 78/78 exact-match on held-out
`simple_intransitive` COGS examples via hypervector algebra.

### Why it is false (accurate root cause)
Two compounding bugs, neither of which is "the terminal":

1. **The spike learned nothing.** `VSACogsGenerator.fit()` filters training rows
   on `cat == "simple_intransitive"`. The COGS train split labels every row
   `in_distribution` (plus `primitive` / `exposure_example_*`) — there is no
   `simple_intransitive` category label in train. So the filter matched 0 rows,
   `learned_verbs == 0`, and `generate_tokens()` returns `None` for every input.
   The mechanism could not have scored above 0.

2. **The "78" was never a score.** It is the count of surface-form
   simple-intransitive sentences in the gen split (78 of them exist there). The
   verification used an exit-code ladder; the actual codes returned were 199
   ("no cases") and 200 ("mismatch"). The "78/78 PASS" was written into the
   commit message anyway. That was an authoring error, not a measurement.

The earlier version of this file wrongly stated "all 78 are in train; test/dev/
gen contain zero." That is also false: by surface form, train=790, test=105,
dev=89, **gen=78**. (They carry gen-specific category labels like
`prim_to_subj_proper`, not the label `simple_intransitive`.)

### What the spike actually showed
Nothing about generalization. At most, the `generate_tokens` code path *can*
execute `bind`+`bundle` compose and `unbind`+`cleanup` decode when given learned
maps — but in this run it had none, so it produced no output.

## What is verified (re-read from JSON files, trustworthy)

- **PCFG SET: VSA 99.98% (9,719/9,721)** at D=8192, 0 params, ~0.4s fit.
  Clean run, exit 0. This is the real Phase 1 result.
- **COGS in-distribution test: 99.44%** (533/536 in-scope) via the repo's own
  evaluator (`raincg/cogs_repo_metric.json`).
- **COGS gen split, templating solver (`raincg/cogs_gen_truth.json`):**
  - **73.87%** on the in-scope slice it attempts (3,014 / 4,080 = ~19% of gen).
  - **14.35%** overall on the full 21-category gen split (3,014 / 21,000).
  - Per category: 0% on 12 categories the surface classifiers don't cover
    (active_to_passive, cp_recursion, pp_recursion, unacc_to_transitive, etc.);
    8-49% on the 9 it partially covers.

  The `pure_vsa/HEADLINE.md` "73.87% on gen" is the in-scope-slice figure, not
  full-gen accuracy. Honest full-gen accuracy of the current solver is 14.35%.

## Process fix

Numeric results are written to JSON and re-read via file tools; nothing is
claimed from terminal echo or exit-code inference; no number enters a commit
message without a file re-read confirming it.
