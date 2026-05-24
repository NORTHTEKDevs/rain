# RAIN -- Research Log

> Where we record HYPOTHESES, EXPERIMENTS, and RESULTS for the
> Phase-2 architectural research program -- novel non-Transformer
> generative architectures built on RAIN's VSA substrate.
>
> Negative results stay logged. The point is to not relearn the
> same failures.

## R&D Program Goal

Build a non-Transformer generative architecture, native to RAIN's
hyperdimensional substrate, that competes with Mamba / RWKV / xLSTM
at small-to-medium scale and exceeds them at large scale by leveraging
properties their architectures lack:

- O(D) bind operations instead of O(D^2) dense matmul
- Bipolar discrete state (hardware-friendly for non-GPU accelerators)
- KB-composable (state lives in the same space as KB entries)
- Continual-learnable codebook (RAIN's `tell()` updates the substrate)

Honest probability of program success in 3 years: 5-15%.
Honest probability in 5 years with funding: 20-40%.
Status: month 0 / day 1.

---

## Experiment 1 -- VSA-Seq v1

**Hypothesis:** A sequence engine where recurrence operates via
bind/bundle/permute on bipolar 10K-D hypervectors (instead of MLP-with-
carry or attention) can match or beat HYMN's baseline NLL on Tiny
Shakespeare while using ~10x fewer learnable parameters.

**Architecture (rain/core/vsa_seq.py):**
- Bipolar codebook (fixed buffer, warm-started via feature method)
- State: float32 vector in [-1, +1]^D via tanh relaxation
- Recurrence: `state_{t+1} = tanh(multilane_permute(state_t) + W_in @ token_hv_t)`
- Multilane permute: L fixed random permutations applied, weighted by
  softmax(lane_weights), summed
- Output: cosine(W_out @ state_t, codebook) * learnable_temp + token_bias
- Trainable params at dim=1024, vocab=65, lanes=4: **2.1M**
  (HYMN equivalent: ~1.5M; comparable)

**Setup:** Tiny Shakespeare, 3K steps, carry=16, batch=16, dim=1024,
lr=1e-3 with 200-step warmup + cosine decay, warm-start-chars init.
Same recipe as the HYMN 3K-step baseline (which hits ~2.4 NLL).

**Result:** **FAILED to learn.** Three readout variants tried:

| Variant | Output head | Predict from state | Final NLL (3K steps) | vs HYMN baseline (2.40) |
|---|---|---|---|---|
| v1 cosine | cos * sqrt(D), no temp | after bundling token | **8.00** | 3.3x worse |
| v2 dot | unnormalized proj @ cb.T / sqrt(D) | after bundling token | **39.4** (diverged) | 16x worse, diverged |
| v3 cosine+temp+bias+causal | cos * learnable_temp + token_bias | BEFORE bundling token | **14.5** | 6x worse, regressed |

Uniform baseline for the 65-char vocab is ln(65) = 4.17. **All variants
underperformed even uniform random prediction at the end of training.**

**Diagnosis (best-guess, not validated):**

1. **Cosine readout is gradient-hostile.** When projected state is unit-
   normalized against a bipolar codebook of {-1, +1}^D, the cosine
   landscape is very flat almost everywhere -- gradients only matter at
   sharp decision boundaries. The model can't smoothly learn.

2. **The bipolar relaxation (tanh) saturates fast.** Over 16 carry steps,
   the bundled state values approach ±1 and lose information. There's
   no normalization or gating like in LSTM/Mamba/RWKV.

3. **Predicting-after-bundle (v1, v2) wastes most signal.** The state is
   dominated by the just-bundled current token, so logits trivially
   predict the current char -- gradient wants to undo the perfect
   identity, fighting itself.

4. **Predicting-before-bundle (v3) is the right structure but the
   recurrence doesn't carry enough next-token signal.** The state never
   gets to encode "the next char given the past" -- it just encodes
   "the past."

5. **Learnable temperature + token bias don't help when the recurrence
   itself isn't producing predictive states.** They become the only
   learning signal; the recurrence is dead weight.

**Honest conclusion:** The prototype as designed has a fundamental
mismatch between the recurrence dynamics (bipolar bind-bundle) and the
prediction task (predict the next discrete symbol). The architecture
needs to be rethought, not just retuned.

**Cost so far:** ~3 hours of session time, ~10 minutes of training compute.
**Cost saved by killing now vs scaling up:** weeks of compute.

---

## Hypotheses for the Next Iteration

Three architectural directions to try next, ordered by my current best
guess at probability of success:

### H1: Gated / multiplicative recurrence (Mamba-style for VSA)

Add an input-dependent gate to the recurrence:

    g_t = sigmoid(W_g @ token_hv_t)
    state_{t+1} = g_t * tanh(multilane_permute(state_t)) + (1 - g_t) * W_in @ token_hv_t

This gives the model the same "selective state" mechanism that makes
Mamba work. Still O(D) operations. Still bipolar-friendly (with sign()
at inference).

**Probability of working:** moderate (30-50%). Mamba's selective scan
is its key innovation; ablating it kills it. A VSA-native gated
recurrence might similarly unlock learning.

### H2: Separate prediction head from state evolution

Two states maintained:
- `s_context`: accumulates the past (bind-bundle as in v1)
- `s_predict = W_pred @ s_context`: a learned "what comes next given
  the context" projection

Logits are computed from `s_predict`, not `s_context`. This separates
the two roles cleanly so the recurrence isn't asked to be both
context-encoder and predictor.

**Probability of working:** moderate (40-60%). This is basically the
LSTM trick (hidden state vs cell state) but for VSA. Conceptually
clean. Would need careful init.

### H3: Drop bipolar relaxation; learn directly in float space

If the bipolar constraint is what's killing gradient flow, just train
in float and "snap to bipolar" only at inference for the discrete-
hardware story. This is the most pragmatic path -- gives up the
"bipolar all the way" purity but keeps the VSA bind-bundle structure.

**Probability of working:** high (60-70%). But it's also the least
architecturally novel, so the moat-vs-Mamba story weakens.

---

## Where this leaves the program

**Day-1 finding:** VSA-Seq is not a slam-dunk. The naive composition
of bind/bundle/permute does not just "work" as a sequence model.

**What this tells us:**
- The 1-5% probability estimate for the multi-year research program
  was reasonable, not pessimistic.
- The next architectural iteration must address the predictive-signal
  bottleneck, not just retune.
- This is the research process. Every "novel architecture" lab has
  notebooks full of failed first attempts.

**What to do now (priority order):**
1. Commit this research log and the prototype code so future-us can
   pick up where day-1 left off (DO).
2. Continue Path-3 product work in parallel (RAIN-on-Llama for
   a vertical) so revenue funds research time.
3. When ready to resume R&D, start from H2 (separate state-vs-
   prediction) -- highest leverage architectural change.
4. Allocate at most one focused day per week to architecture R&D
   until either (a) we find a learning signal worth scaling, or
   (b) we have product revenue to fund full-time research.

**Honest assessment:** the user's instinct that RAIN's VSA substrate
could be the basis of a competitive non-Transformer architecture is
not wrong. But the path from "we have VSA primitives" to "we beat
Mamba" requires real architectural insight that day-1 has not yet
produced. The iterations above might find it; might not.

Don't bet the company on this. Bet the architecture LAB on it, fund
the lab from product revenue, and let it cook for 1-3 years before
declaring victory or defeat.
