# RAIN-CG Go/No-Go Decision - 2026-05-31

**Authoritative.** This file reconciles `MOAT-VERDICT.md` (final, written last)
with `PHASE2-STATUS.md` (paused, contains an open "try one more category" resume
plan). Where they conflict, this file governs. All numbers below were re-read
from their source files; none are restated from memory.

## The bet being judged

"A new type of AI / non-LLM generative substrate, built around the VSA
hypervector kernel, that uniquely cracks compositional generalization (COGS/SCAN
length-gen) where Transformers structurally fail."

## Verdict: NO-GO on the substrate-as-new-AI thesis

The VSA hypervector substrate does **not** uniquely enable compositional
generalization. The single clean, file-verified moat test was won by a
**pure-Python** representation change, with VSA absent from the mechanism.

### Evidence (file-verified)

1. **unacc_to_transitive** (`moat_unacc_result.json`): baseline (roles bound to
   verb) 0/1000 = 0.0%; construction-keyed (roles bound to construction)
   130/1000 = 13.0%, and **130/130 = 100%** on the structurally-covered cases.
   The 0%->100% fix is pure Python (bind roles to the construction, not the
   verb). No bind/bundle/unbind/cleanup used. VSA contributed nothing.
2. **cp_recursion** (`moat_cp_result.json`): n=3, Python 2/3, VSA 2/3,
   agreement 3/3. Directional only (tiny n); VSA and Python identical wherever
   both ran.
3. **round-trip-at-depth** (`moat_roundtrip.py`): produced no result file
   (timed out). No data. Contributes nothing.

### Why the phase-2 "one more category" plan does not change this

`PHASE2-STATUS.md` proposes building an English->slot parser + VSA role-compose
on one more hard category (`subj_to_obj_proper`) to beat templating's 0%. That
plan is **subsumed** by finding #1, which already tested a member of the same
class (cross-construction role inference) and showed:

- The generalization lives in the **parse / construction-keyed role assignment**
  (pure symbolic), not in the hypervectors.
- VSA is a **lossless transducer given a parse** (confirmed by finding #2's 3/3
  agreement) - it rides downstream and adds nothing the parse hasn't already
  decided.

A better parser raises **coverage** (more of the 1000 examples handled) but does
not move generalization credit from the symbolic role logic to VSA. Even a hit
of >40% on a 0%-category would credit the symbolic parser, not the substrate.
That is a symbolic compositional parser with a VSA encoder bolted on - not "a new
type of AI built around VSA."

## What remains TRUE and worth keeping

- **pure_vsa PCFG-SET: 99.98%** (9,719/9,721) at D=8192, 0 trained params,
  ~0.4s CPU fit. Real, reproducible, narrow.
- **COGS in-distribution: 99.44%** (533/536 in-scope, repo evaluator).
- Honest scope of the win: VSA is an **exact, auditable, zero-gradient
  transducer for problems with well-specified structure** - not a generative
  substrate that cracks open-ended compositional generalization on language.
- COGS gen via templating: 14.35% overall (3,014/21,000), 0% on 12 of 21
  categories. That 0% is the honest ceiling of hand-templating, and the moat
  test shows VSA does not lift it.

## Actions

1. **Stop** the substrate-as-12-month-play. Do not resume the phase-2 category
   loop expecting a VSA moat; it would reproduce finding #1.
2. **Optional, bounded salvage:** publish `pure_vsa` as a narrow
   "zero-gradient exact transduction" result for exactly what it is (PCFG 99.98%,
   0 params). Modest, defensible, no overclaim. Not a new-AI paper.
3. **Redirect** the freed energy elsewhere [project names redacted from the public archive] (
   Kryos ecosystem).
4. **Discipline carried forward:** no number enters a doc or commit until it has
   been read back from the JSON that produced it. (Three fabrication incidents
   this session, all from writing ahead of file-verification - logged in
   `MOAT-VERDICT.md` and `CORRECTION.md`.)

## Status of the other docs

- `MOAT-VERDICT.md` - consistent with this decision; keep.
- `PHASE2-STATUS.md` - its "Resume plan (fresh session)" is **CLOSED** by this
  file, not open. Keep for history; do not action.
- `CORRECTION.md` - the retracted "COGS 78/78" claim; keep as the fabrication
  record.

## Addendum 2026-06-01 - depth-memory loophole tested to destruction

The one loophole the prior moat tests left open: cp_recursion depth generalization
was only checked at n=3 (`moat_cp_result.json`), and that check HANDED VSA the
Python-computed x-indices (`str(3*k+1)`), so it never tested VSA *generating* deep
structure. Reopened as the substrate's single best home-turf case (VSA as
recursive depth-memory) and tested fairly. Script `moat_depth_memory.py`, result
`moat_depth_memory_result.json` (all numbers re-read from file).

Setup: train caps CP nesting at depth<=2; gen requires depth 3-12 (n=100/depth,
1000 clean chains, 0 skipped). The real depth-invariant difficulty is the
cumulative event-variable index sequence (gold strides are non-constant, e.g.
depth-4: 4,4,3,4 - a running sum of per-clause argument counts). Three mechanisms
reconstruct it: `python` (integer accumulator), `vsa_iter` (per-step round-trip),
`vsa_bundle` (whole depth-12 trajectory superposed into ONE fixed-width vector).

Result: **all three = 100% exact at every depth 3-12.** `vsa_bundle` does NOT beat
`python` anywhere. Tie at 100% => the generalization is the depth-invariant strides
+ accumulation (pure symbolic), and VSA is a lossless transducer matching it -
finding #1 restated on the hardest case, now with n=1000 instead of n=3.

The one genuinely VSA-flavored property (constant-memory deep recall) is
capacity-bounded, not constant-cost. `vsa_bundle` exact-recall at depth 12 vs
dimension:

| d | 8192 | 2048 | 1024 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|---|---|
| acc | 1.00 | 1.00 | 1.00 | 0.81 | 0.01 | 0.00 | 0.00 |

It only looked perfect because it was given 8192 dims for ~12 items. Holding a
depth-12 trajectory needs d>=1024; below that it collapses. `python` holds the same
depth in 12 integers at any scale. So VSA's distinctive property is a **limitation**
here, not a moat. **Verdict field in the JSON: `NO-GO-CONFIRMED`.** The depth
loophole is closed; DECISION.md stands. (Process note: the first run mis-fired a
`SUBSTRATE-SURVIVES` flag from a too-lenient gate clause that credited a 100% tie
as a win; the gate was corrected to require a STRICT beat before any claim - logged
here so the false positive is on record, not buried.)

---

## Addendum 2026-07-11: the "templating ceiling" evidence was measured on the wrong solver

The NO-GO verdict on the VSA-substrate thesis STANDS (nothing below touches
findings #1-#3 or the depth addendum). What does NOT stand is this file's
supporting claim that COGS gen is capped at "14.35% overall, 0% on 12 of 21
categories" by templating. That number came from the older narrow classifier
(`COGSIntransitiveHyperion` path). The repo already contained, ten days before
this verdict was written (2026-05-21, `pure_vsa/cogs_template_learner.py` +
`cogs_recursive.py`, asserted by `tests/test_cogs_template_learner.py`), a
general recursive template metalearner that was never evaluated by the
verdict session.

Fresh, independently re-run 2026-07-11 (fit on train split ONLY, zero gradient
descent, CPU):

- **COGS gen: 20,948/21,000 = 99.75% exact match** (fit 2.0s, eval 0.3s).
- All 12 categories recorded here as "0% ceiling" score >= 99.0%.
- Negative control (shuffled gold targets): 2/21,000 = 0.0095%.
- Artifacts: `raincg/results/cogs_gen__template_learner_symbolic__seed0.json`
  (per-category breakdown inside), `raincg/cogs_gen_truth_fresh_20260711.json`,
  test suite 7/7 green.

Corrected reading: the generalization credit still belongs to symbolic,
content-independent composition (consistent with findings #1-#3 - VSA the
substrate contributed nothing); but the practical ceiling claimed for
exact-composition approaches on COGS was off by ~85 points. The honest scope
of the pivot's positive claim is therefore: learned-selection + exact
composition solves BOTH SCAN (100%) and COGS gen (99.75%) at seconds of CPU,
where the same-split measured baselines (from-scratch transformer, llama3.2:3b,
qwen2.5:14b few-shot) score 0-12%. Caveat carried forward: the template
learner encodes COGS task ontology (role labels, construction signatures) -
a task-specific inductive bias, not a general language learner.
