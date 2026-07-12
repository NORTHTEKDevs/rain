# COGS Attack Design — 2026-07-11

**Status: design-only. No training run in this pass. Every number below is
read from a JSON file written by an executed script in this session; paths
given inline. Two new artifacts: `raincg/cogs_gen_truth_fresh_20260711.json`,
`raincg/cogs_intrans_role_ablation_20260711.json`.**

## 0. Critical finding first: the premise in the brief is stale

The brief states hand-templating scores **0% on 12 of 21 COGS gen
categories** (citing `raincg/cogs_gen_truth.json`, 14.35% overall) and asks
for a mechanism to beat that 0%. I re-ran the evaluation fresh before
designing anything, per the project's own discipline (no number without its
file). Result:

- `raincg/cogs_gen_truth.json` (14.35% overall, 0% on 12 categories) is
  reproduced exactly by running the OLD narrow classifier
  (`pure_vsa.cogs_hyperion.COGSIntransitiveHyperion`, 6 hand-written
  surface-signature functions) — confirmed byte-for-byte
  (`in_scope=4080, acc=0.7387..., overall=0.14352...`, matches the stale
  file exactly).
- The repo also contains a **second, more general solver**
  (`pure_vsa/cogs_template_learner.py` + `pure_vsa/cogs_recursive.py`, a
  recursive-descent parser over NP-chains, clauses, passive, datives,
  CP/PP recursion, and control) that was **already sitting in the repo,
  dated 2026-05-21** — 10 days before `DECISION.md` (2026-05-31/06-01).
  Fresh run today: **99.75% overall accuracy, 99.94% coverage, on the same
  21000-example gen split** (`raincg/cogs_gen_truth_fresh_20260711.json`).
  Every one of the 12 categories `DECISION.md` calls a "0% ceiling" scores
  **99–100%** under this solver. `tests/test_cogs_template_learner.py`
  (also dated 2026-05-21) already asserts this in code
  (`test_template_learner_specific_gen_categories`, >=95% per category)
  and was presumably green the whole time.
- Spot-checked exact-match predictions by hand on 5 examples across
  `unacc_to_transitive`, `only_seen_as_transitive_subj_as_unacc_subj`,
  `active_to_passive`, `cp_recursion` (including a 10-clause-deep nested
  sentence), and `subj_to_obj_proper` — all 5 matched gold token-for-token.
  No leakage: `fit()` is called on `train.tsv` only, `predict()` never
  touches gold at inference.

**Conclusion:** `DECISION.md`'s "0% on 12 hard categories" describes a
solver that was already superseded in this repo before `DECISION.md` was
written. The session that wrote `DECISION.md`/`PHASE2-STATUS.md` evaluated
the old classifier and never found the newer recursive parser. The "beat
0%" framing this task was scoped against does not describe the current
repo. This is not a disagreement with `DECISION.md`'s VSA verdict (that
verdict is untouched — see §4) — it is a stale-baseline bug in the premise
of *this* task, caught by re-executing before designing, per the project's
own standing rule.

**What this means for the deliverable below:** there is no "0%-category"
left to attack — coverage/accuracy on COGS gen is closed (99.75%, pure
Python, zero VSA, zero neural nets, zero gradient descent) by code that
already exists. The five sections below still answer the brief's actual
underlying question — does the F22 output-only-REINFORCE-+-exact-executor
*pattern* generalize past SCAN — but retargeted at the one place that
question is still open in this repo, found by ablation (§2), not invented.

## 1. Mechanism design

**What's exact (unchanged, keep as-is):** the recursive-descent structural
parser in `pure_vsa/cogs_recursive.py` — NP-chain parsing, clause/CP/PP
recursion, passive/dative/control detection, and **all position-keyed role
assignment** (subject-of-active-transitive → agent, object → theme,
by-phrase → agent, recipient → recipient, etc., all hardcoded literals, no
per-verb lookup). This is a hand-written but *correct* reconstruction of
COGS's actual generation grammar — role assignment in COGS is overwhelmingly
determined by syntactic position, not verb identity, which is why a
position-keyed parser gets 99%+ without learning anything. This is the
"exact composition" layer, structurally the same role `assemble` +
`expected_output` play in `experiments/scan_hybrid.py`.

**What's currently a supervision cheat, not a learned mechanism:** exactly
one artifact, `COGSTemplateLearner.intrans_role: dict[verb_inf -> 'agent'|
'theme']` (61 entries, built by `pure_vsa/cogs_recursive.py:learn_intrans_role`).
This function does not learn from output-only reward — it **scans gold
training logical forms for the literal tokens `agent`/`theme` next to each
verb** and copies them into a table. That is supervised extraction of a
structural label, the same category of shortcut Finding 21 used before F22
replaced it on SCAN. It is the one non-positional, per-lexical-item
decision left in the whole pipeline (verified by ablation, §2) — whether an
intransitive verb is unaccusative (theme subject: "the vase shattered") or
unergative (agent subject: "the dog barked"), a real fact about English verb
classes that COGS's grammar does encode lexically, not positionally.

**Proposed mechanism (mirrors F22 exactly, applied to this one decision):**
replace `learn_intrans_role`'s gold-label scan with a per-verb-type logit
table `theta: torch.zeros(V, 2)` (V ≈ 61, one row per infinitive verb type),
trained by REINFORCE against the *same* exact executor already in the repo
(`predict_recursive` / `emit_intransitive_output`), with reward computed
**only** from whether the fully assembled output string exact-matches gold
— the learner never reads a role-label token, only a scalar. This is
`experiments/scan_hybrid_outputonly.py`'s recipe verbatim: sample →
assemble via the exact executor → shaped reward → REINFORCE with a moving
baseline. Supervision signal: **output-only REINFORCE**, not supervised
parses, not hand labels — matching F22's discipline, not Finding 21's.

Credit split, stated explicitly per the corrected thesis: the parser
(exact, hand-written, ~800 lines of Python) does the composition and covers
99%+ of the benchmark on its own with zero learning. The learned piece is a
122-parameter (61×2) table covering one binary lexical fact. **No VSA
anywhere in this pipeline.** The claim under test is narrower than the
brief originally posed: not "can learning beat a 0% wall" (there is no
wall) but "can the F22 output-only-REINFORCE recipe replace a supervised
shortcut with something that never reads structural labels, and hold the
same accuracy." That is still a legitimate, if smaller, generalization
test of the mechanism.

## 2. The minimal spike

**Category (chosen by ablation, not by the brief's suggestion):** I zeroed
`intrans_role` (forced every verb to the majority-class default "agent")
and re-ran the full 21000-example gen eval
(`raincg/cogs_intrans_role_ablation_20260711.json`). Every category is
unaffected except two:

| Category | With table (cheat) | Table zeroed (floor) |
|---|---|---|
| `only_seen_as_transitive_subj_as_unacc_subj` | 994/1000 = 99.4% | **44/1000 = 4.4%** |
| `cp_recursion` | ~1000/1000 ≈ 100% | 689/1000 = 68.9% |
| everything else (19 categories) | 99–100% | 99–100% (unaffected) |
| overall gen (21000) | 99.75% | 93.35% |

`subj_to_obj_proper` and PHASE2-STATUS's suggested `subj_to_obj_proper`
target are **not** gated by this decision at all (unaffected by the
ablation, 100% either way) — that category's earlier suggestion in
`PHASE2-STATUS.md` predates the recursive parser and is moot. The one
category genuinely gated by a non-positional, per-verb-type decision is
`only_seen_as_transitive_subj_as_unacc_subj`. `cp_recursion` is a free
cross-category validation of the same learned table (same verb types recur
inside CP-embedded clauses), used as a secondary metric.

**Why this category, concretely:** it draws on 20 distinct embedded verb
types (`roll, freeze, burn, shorten, float, slide, grow, crumple, change,
double, ...` — classic causative-inchoative/unaccusative verbs), all 20
already covered by the existing (cheat) table
(`raincg/cogs_intrans_role_ablation_20260711.json` →
`target_category_only_seen_as_transitive_subj_as_unacc_subj`). With the
table zeroed, the default-agent floor gets 44/1000 right (cases where
agent happens to be correct by chance/other structure) — this is the real,
freshly-measured "0%-analog" baseline, not a fabricated 0%.

**Train pool:** `is_simple_intransitive` (3-token "ProperNoun VerbPast .")
examples in `train.tsv` — **790 examples, 61 distinct verb types**, ~13
examples/verb on average, **zero label-inconsistent verbs** (every verb has
one fixed gold role across every training occurrence — a clean,
noiseless per-verb binary bandit, confirmed in
`raincg/cogs_intrans_role_ablation_20260711.json` →
`reinforce_training_pool_bare_intransitive_train`).

**Exact protocol:**
1. `theta = torch.zeros(61, 2, requires_grad=True)`, `opt = Adam(lr=0.05)` —
   same hyperparameters as `scan_hybrid_outputonly.py`.
2. Per step: sample batch of B=64 from the 790 bare-intransitive train
   examples; for each, `role ~ Categorical(softmax(theta[verb_id]))`;
   assemble via `emit_intransitive_output(verb_inf, role_name, pname)`
   (already in `pure_vsa/cogs_hyperion.py`, unchanged); reward =
   `shaped_reward(pred, gold)` from `scan_hybrid_outputonly.py` verbatim
   (1.0 exact match, else 0.4× token-overlap); REINFORCE with a 0.95-decay
   moving-average baseline.
3. Steps: 300 (generous — 61 independent 2-way bandits with clean signal
   should converge in tens of steps; run to 300 for margin, checkpoint
   every 25).
4. Freeze `theta`, take `argmax` per verb → a learned `intrans_role` dict.
   Substitute it into `COGSTemplateLearner.intrans_role` in place of the
   gold-scanned one (one-line swap; nothing else in the pipeline changes).
5. Re-run `coverage_stats()` on the full gen split, exactly the machinery
   already used for `raincg/cogs_gen_truth_fresh_20260711.json`. Report,
   to JSON, re-read before any claim: (a) accuracy on
   `only_seen_as_transitive_subj_as_unacc_subj`, (b) accuracy on
   `cp_recursion`, (c) overall 21000-example gen accuracy.

**Baseline to beat:** 4.4% (measured, not assumed) on the target category.

**Acceptance threshold:** two bars, stated honestly against the real
number, not the brief's assumed 0%:
- **Floor (matches the brief's original ">40% on a 0% category" intent,
  corrected against the real baseline):** clear 40 percentage points above
  the measured 4.4% floor → **≥44.4%** absolute on
  `only_seen_as_transitive_subj_as_unacc_subj`.
- **Real bar (what actually matters):** recover ≥90% of the 99.4% cheat
  ceiling → **≥89.5%** absolute, i.e. output-only REINFORCE nearly matches
  what gold-label-scanning already achieves, closing the "is this really
  output-only" discipline gap without accuracy loss.

**Negative controls (falsification, mandatory before any positive claim):**
- **Shuffled-target control:** run the identical REINFORCE loop but pair
  each training input with a randomly permuted (wrong) gold output within
  the batch. Expect it to fail to clear the 44.4% floor — if it does clear
  it, the reward computation is buggy or leaking, and the positive result
  is void regardless of what it shows.
- **Label-flip control:** invert the reward (reward = 1 when sampled role
  **≠** gold role). Expect the learned table to converge to the exact
  opposite per-verb classification from the honest run, scoring
  symmetrically high against the *flipped* metric and correspondingly near
  the 4.4% floor against true gold. This proves the mechanism is extracting
  the target's real signal, not defaulting to a fixed heuristic that scores
  well independent of what the reward actually says.
- Both controls' results go to JSON next to the positive run's, all three
  re-read from file before the spike is written up as pass/fail.

**Estimated CPU runtime:** `fit()` on the full template learner: 2.4s
(measured, `raincg/cogs_gen_truth_fresh_20260711.json` run). REINFORCE loop:
122-parameter table, batch 64, 300 steps of pure tensor ops — low single-digit
seconds. Full spike including both negative controls and the gen re-eval:
**under 2 minutes wall time**, no GPU.

## 3. Kill criteria

- Positive run fails to clear 44.4% after 300 steps: stop, do not extend
  the step budget past one retry (per project debugging discipline — two
  iterations max on the same failure, then rethink or report honestly).
  Likely cause if this happens: a wiring bug (reward not reaching the
  sampled role, or `emit_intransitive_output` called with the wrong
  argument order) — check by hand on 3 examples before touching
  hyperparameters.
- Either negative control clears the floor: **stop, discard the positive
  result regardless of its own score.** This is the exact failure mode
  logged in `raincg/CORRECTION.md` and `raincg/MOAT-VERDICT.md` (numbers
  reported before verification) — a leaking or fabricated positive must
  never reach a doc or commit.
- Positive run clears 44.4% but not 89.5%: report the honest number as the
  result — this is still a legitimate, reportable finding (output-only
  REINFORCE partially recovers the decision), just not full parity with
  the supervised version. No overclaim either direction.
- Given the ablation already shows a clean, label-consistent, small
  (61-way binary) bandit problem, I do not expect a kill scenario to
  trigger — see probability estimate below. Stating this plainly rather
  than hedging: this spike is close to a foregone conclusion on the
  positive side, and the real risk is entirely in the negative controls
  (proving the win isn't a wiring artifact), not in whether REINFORCE
  converges.

## 4. How this answers DECISION.md's objection

`DECISION.md` predicted exactly this outcome and pre-emptively ruled out
crediting VSA for it: *"A better parser raises coverage... but does not
move generalization credit from the symbolic role logic to VSA. Even a hit
of >40% on a 0%-category would credit the symbolic parser, not the
substrate."* That prediction is now confirmed twice over — first by
`cogs_recursive.py` itself (a better parser, 100% Python, raised coverage
from 19% to 99.94% and accuracy from 14.35% to 99.75%, with zero VSA in the
mechanism), and second by this spike's design, which credits an F22-style
output-only REINFORCE table (61×2 parameters, no hypervectors, no
bind/bundle/unbind) for the one remaining decision. Under the corrected
thesis this is not a problem — the research claim was never "VSA uniquely
generalizes," it's "learned selection + exact symbolic composition
generalizes as a pattern," and VSA is explicitly absent from both the
parser and the learned component here, same as it was absent from
`moat_unacc_to_transitive.py`'s finding and from `scan_hybrid_outputonly.py`.
`DECISION.md`'s NO-GO on the *substrate* thesis stands untouched; this
design provides zero evidence for or against VSA either way, by
construction.

## 5. Estimated effort to the full result

**To the original ask (COGS's 12 hard gen categories beating a 0%
baseline): zero additional effort.** It is already done, in the repo,
dated 2026-05-21, pre-dating this task by seven weeks — 99.75% overall,
99–100% on every one of the 12 categories `DECISION.md` called a 0%
ceiling (`raincg/cogs_gen_truth_fresh_20260711.json`,
`tests/test_cogs_template_learner.py`). No spike, no new code, no training
run closes a gap that isn't there.

**To the real remaining gap (§2's spike — replacing the one
gold-label-scanning shortcut with genuine output-only REINFORCE):**
0.5–1 day. `experiments/scan_hybrid_outputonly.py` is a near-verbatim
template (swap the per-token 5-way table for a per-verb-type 2-way table,
swap `assemble` for `emit_intransitive_output`, swap the SCAN split loader
for the `is_simple_intransitive` train-pool extractor); the exact executor,
reward shaping, and REINFORCE loop transfer unchanged. Most of the time is
the two negative controls and JSON-verified write-up, not the RL itself.

**Recommended follow-up outside this design (not executed here, DESIGN-ONLY
pass):** `raincg/DECISION.md` and `raincg/PHASE2-STATUS.md` are marked
"authoritative" and currently assert a false baseline (14.35% / 0% on 12
categories) for the present state of the repo. They should be amended with
a dated addendum pointing to `raincg/cogs_gen_truth_fresh_20260711.json`
and this file, the same way `DECISION.md` already carries its 2026-06-01
depth-memory addendum — not rewritten, appended, preserving the fabrication
history as a record per `CORRECTION.md`'s own precedent. Two smaller,
optional follow-ons if the discipline gap is worth closing fully:
`learn_ditrans_verbs` (dative-verb detection) has the same gold-label-scan
shortcut and would take the same ~0.5 day to convert; I did not ablate it
here because ablating it wasn't needed to answer this task, and it should
get its own measured baseline before anyone designs against it — the exact
mistake this task's premise made.

## Probability estimate

**≥95% probability the positive spike clears the 44.4% floor. ≥80%
probability it clears the real 89.5% bar.** This is a small (61-way),
label-consistent, noiseless-reward binary bandit problem, not a hard
sequential credit-assignment problem like SCAN's per-token tagging — the
main real risk is entirely in the negative controls (confirming the win
isn't a wiring artifact) and in engineering time, not in whether REINFORCE
converges. This should not be read as a strong test of the learned-selection
+ exact-composition pattern's power; it is a clean, honest, small
confirmation that the pattern transfers past SCAN, on the one place in
COGS gen that genuinely required it.
