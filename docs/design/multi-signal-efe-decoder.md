# Multi-Signal Expected Free Energy Decoder

**Companion to:** `docs/plans/2026-05-22-rain-design.md` (§ 3 step 5-6)
**Status:** Design sketch. RCK's existing 4-source EFE decoder
(`rck/efe.py`) is extended to 7 sources for RAIN. Empirical validation is
benchmark A1 (`evals/tier3_soundness/efe_source_mix.py`).

---

## The decoder's job

At step `t`, given the current state `s_t`, the predicted next-state
`s_{t+1}` from HYMN, the routed bands `s_t^{L,M,H}`, and the broader context
(KB, crystals, ToM, self-model, calibration tally), produce a distribution
`p(o_{t+1} | s_{t+1}, context)` over the vocabulary and sample one token
(or, for tool-use head, one action).

This is `argmin_π G(π)` from the FEP-unified objective, computed
approximately via candidate generation + weighted fusion + calibration
re-weight + anti-repetition penalty + top-p sample.

---

## The seven candidate sources

For each source, the candidate generator returns a list of
`(token_id, score)` pairs. Total unique candidates typically 50-200 per step.

### Source 1 — HYMN prediction

```
s_{t+1} = HYMN(s_t, o_t)
scores_HYMN = cos_sim(s_{t+1}, C)  # (V,) cosine to each codebook vector
candidates_HYMN = top_k(scores_HYMN, k=64)
```

Weight default: 4.0. Represents the amortized posterior's preference.

### Source 2 — LSM recurrent recall

```
candidates_LSM = LSM.predict_next(history)   # RLS-based recurrent prediction
```

Weight default: 2.0. Represents short-term sequence dynamics.

### Source 3 — Bigram VSA n-gram

```
candidates_BIGRAM = bigram_memory.query(o_{t-1}, k=32)
```

Weight default: 0.5. Represents pure n-gram statistics; small weight because
weakly informative but cheap and useful for fluency.

### Source 4 — FEP action selection

```
candidates_FEP = fep.act(s_t, A_current, k=16)
                # argmin G under low-rank generative model
```

Weight default: 1.5. Represents the "what action minimizes expected free
energy" arm. For tool-use head, this source produces action candidates
rather than token candidates.

### Source 5 — Sharded HRR KB lookup

```
query = build_query(s_t, recent_context)   # bind subject + relation roles
candidates_KB = kb.query(query, k=32)      # shard-routed by blake2b
```

Weight default: 3.0. The factual-grounding signal. Closes the "encyclopedia
recall" gap to LLMs. For factual prompts this often dominates.

### Source 6 — Tsetlin clause vote

```
clause_votes = tsetlin.vote(s_t)  # (V,) score per token from clause population
candidates_TSETLIN = top_k(clause_votes, k=32)
```

Weight default: 1.0. Represents the symbolic structural-validity prior.
Critical for code head where syntax matters.

### Source 7 — Crystal recall

```
embedding = embedFn(recent_context)
similar_crystals = polyglot.crystals.search(embedding, k=8)
candidates_CRYSTAL = extract_next_tokens(similar_crystals)
```

Weight default: 2.5. Represents episodic prior; matches recent-conversation
patterns and prior interaction styles.

---

## Fusion

Merge all 7 candidate lists into a per-token score:

```
score(token) = Σ_s  w_s · (rank-normalized score of token in source s)
```

Where `w_s` are the per-source weights (defaults above; NSGA-II evolved at
runtime via the workload observer).

Rank-normalization: convert each source's scores to ranks in `[0, 1]`. This
prevents one source from dominating purely because of score magnitude.

---

## Calibration re-weight

Apply `cognition/metacog.py` per-relation tally:

```
for token in candidates:
    relation = identify_source_relation(token)   # which KB relation produced
                                                  # this candidate
    calibration = metacog.calibration[relation]  # ECE-based confidence weight
    score[token] *= calibration
```

Tokens whose source-relation has historically low calibration (the model
frequently overstates confidence on this relation) get damped. Tokens whose
source-relation has historically high calibration (model is correct when
confident) get amplified.

---

## Anti-repetition penalty

Three windows, all from RCK v1.0:

```
penalty(token) = single_char_penalty(token, last_token)
               + bigram_cycle_penalty(token, last_2)
               + trigram_cycle_penalty(token, last_3)

score(token) -= penalty(token)
```

---

## Top-p / nucleus sampling

Standard:

```
sorted = sort(candidates by score, desc)
cumprob = cumsum(softmax(sorted.scores))
nucleus = sorted[cumprob < p]    # default p = 0.9
sampled = sample_proportional(nucleus.scores)
```

Returns one token + a confidence score (the softmax probability of the
sampled token within the nucleus).

---

## Epistemic class assignment

`cognition/metacog.py` classifies the output:

| Class | Condition |
|---|---|
| `know` | Confidence > τ_know AND relation-calibration > κ_know |
| `think` | τ_think < Confidence ≤ τ_know |
| `guess` | τ_guess < Confidence ≤ τ_think |
| `unknown` | Confidence ≤ τ_guess OR nucleus had < 2 candidates |

Thresholds NSGA-II-evolved. Defaults: `τ_know = 0.85`, `τ_think = 0.50`,
`τ_guess = 0.20`.

`unknown` triggers refusal/hedge phrase + learning mode (current input
becomes a candidate teaching example).

---

## Per-step compute cost

| Source | Cost |
|---|---|
| HYMN | O(D × H + V × D) — dominated by codebook similarity |
| LSM | O(D^2) — RLS forward |
| Bigram | O(log V) — hashed lookup |
| FEP | O(D × R) where R = rank of A |
| KB | O(K × log shard_size) for K shards |
| Tsetlin | O(C × D) where C = clause count, vectorised |
| Crystal | O(D × N_crystal) cosine; HNSW-indexed |
| Fusion | O(N_candidates × log N_candidates) |
| Calibration | O(N_candidates) |
| Anti-repetition | O(N_candidates) |
| Top-p sample | O(N_candidates × log N_candidates) |

Total per token at v0 toy scale: ~5-10 ms CPU (matches RCK v1.0 measured
baseline). With Rust hot path: 1-2 ms.

---

## Speculative-decoding optimization

Capability addition #7 uses bigram + Tsetlin (sources 3 and 6) as fast
proposers, then HYMN + FEP + KB (sources 1, 4, 5) as the verifier:

```
proposed = bigram_and_tsetlin_top_k(s_t, k=4)  # ~0.5 ms
verified = hymn_efe_score(proposed, s_t)        # ~5 ms
return verified.argmax()
```

When the fast proposer's top pick matches the verifier's top pick (~60-80%
of the time on typical text), the verification step short-circuits and the
output ships at proposer speed. 2-4x perceived speedup with no quality
loss because the verifier always has final say.

---

## Tests

- `tests/core/test_efe_fusion.py` — 7-source fusion correctness, rank-normalization
- `tests/core/test_efe_calibration.py` — per-relation re-weighting
- `tests/core/test_efe_anti_repetition.py` — penalty calculations
- `tests/core/test_efe_epistemic_class.py` — class assignment thresholds
- `evals/tier3_soundness/efe_source_mix.py` — A1 benchmark (each source ≥5%)
