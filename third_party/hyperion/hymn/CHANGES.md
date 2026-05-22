# HYMN — changes since 0.1.0

## Phase 1d additions (second session)

### Two new readout heads + KV-write mechanism

- **`ContextConditionedHead`** -- takes the current input token's HV as
  a third argument and uses it to query the field via bind-then-cleanup.
  Identified as the single biggest architectural gap from Phase 1c's
  recall bench. New best on the copy bench: **100% at 150 steps**.
- **`HopfieldCleanup`** -- iterative associative cleanup against the
  codebook (Ramsauer 2020 style). Loses to other heads at small step
  count (55% vs 95%+) -- needs many more iterations or a different
  attention-temperature schedule to be competitive.
- **`HYMNConfig.kv_writes`** -- when True, each input token is bound
  with a learnable per-position key BEFORE being written to the field.
  Field becomes a literal associative memory: `bind(field, key_t)`
  recovers the token at position `t`. Required for any true VSA-style
  recall.

### Empirical findings (recall bench + copy bench)

| readout | copy (150 steps) | recall K=2 (500 steps) |
|---|---|---|
| default | 99.4% | 14% |
| vsa_cleanup | 99.4% | -- |
| bind_query | 93.6% | -- |
| superpos | 95.4% | -- |
| **context_conditioned** | **100.0%** | 15% |
| hopfield | 55.5% | -- |

Honest interpretation:
- Copy task: every readout (except hopfield) learns the trivial task.
  This confirms the plumbing works and gradient flows.
- Recall task: context_conditioned + kv_writes is the architecturally
  correct combination, but 500 steps isn't enough to learn position
  semantics (mapping a position-token to "look up key_i"). Needs
  longer training OR an explicit address-decoder layer.

### Research roadmap document

`design/HYMN-RESEARCH-ROADMAP.md` -- honest, milestone-by-milestone path
from current substrate to a citable "new class of AI" via SCAN
compositional generalization. M1 done; M2 (positional recall) is the
current blocker; M5 (arxiv paper) requires ~2-3 months part-time;
M6 (scale up) requires GPU + external resources.

## Phase 1c additions (first session)

Three new components, all CPU-tractable, all backwards-compatible (default
config behavior is unchanged):

### 1. Swappable readout heads (`hymn/readout.py`)

The flat `field_bundle @ codebook.T` readout in `HYMNMini.output_logits`
discards the VSA inductive bias. Three replacements:

- **`VSACleanupHead`** — same algebra, exposed as a swappable module so
  the variants below can compose.
- **`BindQueryHead`** — learnable query hypervector(s) bound with the
  field bundle before cleanup. Gives the readout compositional structure:
  "what am I looking for?" becomes a learnable VSA primitive.
- **`SuperposCleanup`** — attention-over-codebook + soft VSA cleanup.
  Returns a weighted bundle of candidate tokens rather than collapsing to
  a single token early. Useful when next-token is genuinely ambiguous.

Activate via `HYMNConfig.readout = "bind_query" | "superpos" | "vsa_cleanup"`.
Default behavior preserved with `readout = "default"`.

### 2. Bipolar regularization

Auxiliary loss `mean((1 - h^2)^2)` pushes field cells toward `{-1, +1}`.
The default `tanh` only keeps cells in `[-1, +1]`; this term pushes them
to the *corners*. Encourages crisp VSA states and stops the field from
drifting into dense continuous space where the cosine-sim cleanup loses
its inductive bias.

Activate via `HYMNConfig.bipolar_weight = 0.1` (or whatever weight you
want). Default 0.0 = off.

### 3. Synthetic recall benchmark (`experiments/hymn_recall_bench.py`)

The first iteration target for HYMN improvements. Trains the model to
recall the i-th of K tokens given a position query. Runs in under a
minute per configuration on CPU. Verbose logging shows CE + test-acc
every 50 steps.

**Honest finding from running it:** at K=2, d=128, bind_query, 500 steps
shows only marginal improvement (acc = 0.14 vs random 0.125). The
gradient flows and the loss drops slightly, but the readout being
context-free is the blocker — the model has to learn position semantics
indirectly. This is documented as a known limitation in
`design/HYMN-PHASE3-ACTIVE-INFERENCE.md`. Next architectural move is a
context-conditioned readout (the readout sees the most recent input as
an explicit query key).

### 4. Phase 3 active inference spec (`design/HYMN-PHASE3-ACTIVE-INFERENCE.md`)

The audit explicitly deferred AI from Phase 2. This doc specifies the
*real* minimum-viable variational variant:

- Diagonal-Gaussian posterior `q(h_t | x_<=t) = N(mu_t, sigma_t^2 I)`
- Reparameterized sampling for the CE term
- KL between successive posteriors as the complexity term
- Optional EFE term for genuine exploration
- Three validation criteria the variant must pass to justify its cost

Implementation cost notes are honest: this needs GPU compute to be worth
running. Documented as design-only for now.

## What still doesn't work

- **Recall with positional query.** The bench above shows the architecture
  has a structural gap on this canonical compositional task. Either the
  readout needs to be context-conditioned, or the field-write mechanism
  needs distinct cell-per-timestep encoding rather than the current
  positional permutation.
- **Head-to-head vs NanoGPT at small scale.** Previously documented; still
  true. HYMN-Mini at 12M params loses badly to a tuned NanoGPT at the
  same scale on standard LM benchmarks.

## What does work

- All 29 unit tests pass (23 prior + 6 new for readout heads + 6 for
  bipolar regularization).
- All 4 readout heads forward + backprop correctly.
- Bipolar loss is zero on perfectly bipolar states, positive otherwise.
- The synthetic recall harness is a fast iteration target for future
  changes.

## When to come back to HYMN

Per the project's overall portfolio strategy:
1. After Kryos hits MRR.
2. After a specific research hypothesis presents itself (e.g. "VSAs can
   do task X at lower compute than transformers").
3. After GPU compute is available.
