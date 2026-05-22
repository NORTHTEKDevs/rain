# SOFAR-on-VSA Transplant

**Companion to:** `docs/plans/2026-05-22-rain-design.md` (§ 2, § 3 step 1-3)
**Status:** Design sketch. Full derivation lands during implementation
phase. The math holds in principle; empirical validation is benchmark A2
(`evals/tier3_soundness/sofar_ablation.py`) — if A2 fails, this transplant
is dropped and the architecture proceeds with uniform routing.

---

## What SOFAR does in its original setting (transformers)

SOFAR (NORTHTEKDevs/SOFAR, v0.4.0) is a residual-stream adapter for
transformer language models. Its three load-bearing primitives are:

1. **SVD of read+write matrices** (`sofar/mapper.py`): SVD applied to the
   concatenation of the read matrices (W_q, W_k, W_v, W_c_fc, W_gate, W_up)
   and write matrices (W_o, W_c_proj, W_down) per layer, producing principal
   directions that surface in BOTH what the layer reads from the residual
   stream AND what it writes back. These directions are interpretable as
   "load-bearing semantic axes" for the layer.

2. **Frequency-layered encoder** (`sofar/encoder.py`): the residual stream is
   decomposed into LOW / MID / HIGH frequency bands via FFT-style spectral
   partition, with NaN-safe masking on the low band.

3. **Beam-steering adapter** (`sofar/attention.py`): a LoRA adapter
   parameterized over the SVD directions, with three energy-normalized
   envelope shapes — FOCUS (Gaussian: attend to one direction), SWEEP
   (sinusoidal: scan across directions), TRACK (sigmoidal: follow a chain
   of directions). Energy-normalized to `sqrt(seq_len)`.

In transformers, these primitives adapt the residual stream's information
flow without retraining the base model.

---

## What we're transplanting them onto

RAIN has no transformer. It has:

- A bipolar 10K-dim VSA state `s_t` evolved by HYMN at each step.
- A codebook `C ∈ R^{V × D}` where each token is a hypervector.
- Role hypervectors `R ∈ R^{K × D}` for binding (subject / predicate / etc.).
- The HYMN MLP itself, with weight matrices `W_in ∈ R^{D × H}` and
  `W_out ∈ R^{H × D}`.

The state `s_t` plays the role of the residual stream. The
"read matrices" of RAIN are `(C, R, W_in)` — everything the current step
reads FROM the state. The "write matrices" are `(C^T, R^T, W_out)` —
everything the current step writes TO the state. These are direct analogs.

---

## How each SOFAR primitive maps onto RAIN

### Primitive 1 — SVD on codebook + role + HYMN matrices

`rain/routing/mapper.py` computes:

```
M_read  = concat([C, R, W_in],  axis=0)   # shape (V + K + H, D)
M_write = concat([C.T, R.T, W_out], axis=1)  # shape (D, V + K + H)
M_combined = concat([M_read, M_write.T], axis=0)  # shape ((V+K+H)*2, D)
U, Σ, V_T = svd(M_combined, k=R)   # truncated to top-R routing directions
```

The top-R right singular vectors `V_T` are the routing directions: directions
in the D-dimensional state space that are both heavily read AND heavily
written across the codebook, roles, and HYMN weights.

These directions are cached. Recomputation triggers when:
- The codebook has been updated by Phase-2 PCN local-rule writes by > τ
  fraction of dimensions.
- Roles have been added or modified.
- HYMN weights have been changed (only happens during Phase 1 or during
  champion promotion in NSGA-II).

### Primitive 2 — Frequency-banded decomposition

Bipolar hypervectors are not smooth signals — they live in `{-1, +1}^D`.
Direct FFT spectral partition doesn't apply.

Instead, RAIN uses a **Hadamard partition**: the Walsh-Hadamard transform
of `s_t` gives a spectral representation in a sign-compatible basis. The
spectrum is partitioned into three equal-size bands by frequency rank (low
1/3 of indices = LOW, middle 1/3 = MID, high 1/3 = HIGH). Each band is
zeroed-out outside its range and inverse-transformed back to the original
space, giving `s_t^L`, `s_t^M`, `s_t^H` ∈ `R^D` with `s_t = s_t^L + s_t^M + s_t^H`.

Interpretation under FEP hierarchical generative model:
- `s_t^L` = paragraph-level / slow context drift
- `s_t^M` = sentence-level role bindings
- `s_t^H` = token-level fillers

NaN-safety from SOFAR's `LowBandEncoder` carries over directly.

### Primitive 3 — Beam-steering envelope

For each band `b ∈ {L, M, H}`, choose a beam mode and an envelope:

```
mode = beam_selector(dialogue_state, introspection_state)
                    # returns FOCUS / SWEEP / TRACK

envelope_FOCUS(α)   = exp(-((α - μ)^2) / (2σ^2))         # Gaussian
envelope_SWEEP(α)   = sin(2π · (α - μ) / Λ)               # sinusoidal
envelope_TRACK(α)   = sigmoid(k · (α - μ))                # sigmoidal

# Project state onto routing directions
projections = V_T @ s_t^b              # shape (R,)
weighted    = projections * envelope(arange(R), μ, σ_or_Λ_or_k)

# Re-project back
s_t^b_routed = V_T.T @ weighted        # shape (D,)
```

The envelopes are energy-normalized to keep total signal power constant
(SOFAR's `sqrt(seq_len)` normalization adapted for the projection
dimensionality).

The LoRA adapter is then applied to the routed state via:

```
s_t^b_out = s_t^b_routed + lora_up @ (lora_down @ s_t^b_routed) * beam_gate
```

With `lora_up = 0` and `beam_gate = 0` at init (SOFAR's identity-at-init
guarantee), the entire routing block is the exact identity before any
training step. This is critical for safe integration: enabling routing
never breaks the baseline.

---

## Why this is non-trivial

SOFAR's math was designed for *real-valued, smooth* transformer hidden
states. RAIN's state is *bipolar, sign-compatible*. The Hadamard substitution
for FFT is the central architectural call: it preserves the spectral-
partition intuition in a sign-compatible basis.

If the Hadamard substitution doesn't recover useful frequency structure
empirically (benchmark A2 fails), fall back to uniform routing (no SVD
projection, just identity envelopes) and the rest of the architecture is
unaffected. SOFAR routing is an additive feature, not a load-bearing
dependency.

---

## Open questions

1. **Does the SVD spectrum on bipolar HVs concentrate enough to make
   truncation meaningful?** Empirical question. If the spectrum is flat,
   the top-R directions aren't more informative than random projections and
   routing is no better than uniform.
2. **What is the right beam-mode selector signal?** Current sketch uses
   `dialogue_state + introspection_state`. May need NSGA-II-evolved selector
   policy.
3. **Should band routing be per-step or per-token?** Per-step is cheaper;
   per-token is more expressive. Empirical question.

---

## Tests

- `tests/routing/test_mapper.py` — SVD shape + truncation correctness
- `tests/routing/test_encoder.py` — Hadamard partition round-trip
- `tests/routing/test_beam.py` — identity-at-init holds; envelope energy
  normalization correct
- `evals/tier3_soundness/sofar_ablation.py` — A2 benchmark (val-loss with vs
  without routing); kill criterion if no ≥3% improvement
