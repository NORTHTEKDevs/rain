# FEP-Unified Objective for RAIN

**Companion to:** `docs/plans/2026-05-22-rain-design.md` (§ 4)
**Status:** Draft derivation. Filled in fully during the writing-plans
implementation phase. Reviewers: do not cite as final until peer-reviewed
externally under NDA.

---

## 1. Variational Free Energy

For a generative model `p(o, s)` over observations `o` and latent states `s`,
and an approximate recognition density `q(s)`, the variational free energy is:

```
F[q] = -E_q[log p(o, s)] - H[q(s)]
     = E_q[log q(s)] - E_q[log p(o, s)]
     = E_q[log q(s) - log p(s) - log p(o | s)]
     = KL(q(s) || p(s)) - E_q[log p(o | s)]
```

`F` is an upper bound on `-log p(o)` (surprise). Minimizing `F` w.r.t. `q`
minimizes both the divergence from the prior and the expected surprise under
the generative model.

## 2. Expected Free Energy for action

For a policy `π` (a sequence of actions, including token emissions and tool
calls), the Expected Free Energy is:

```
G(π) = E_q[log q(s | π) - log p(o, s | π)]
     = expected_log_evidence + expected_information_gain
     ≈ -E_q[log p(o | s, π)] + KL(q(s | π) || q(s))
```

Action selection in RAIN is:

```
π* = argmin_π G(π)
```

Computed approximately by the multi-signal EFE decoder over candidate next-
tokens / next-actions.

## 3. Mapping RAIN components to `F` minimization

| Component | What it minimizes / contributes |
|---|---|
| HYMN MLP | `q(s_{t+1} | s_t, o_t)` amortized posterior. Parameters trained Phase-1 to minimize next-token NLL = bound on `-log p(o)`. |
| SOFAR SVD + beam-steering | Precision weighting `Π` on the residual stream; multiplies the prediction-error term in `F`. |
| Frequency-band encoder | Hierarchical factorization `p(s^L) p(s^M | s^L) p(s^H | s^M)`. Each band has its own `F` slice. |
| Sharded HRR KB | Long-term grounded prior `p(s)`. KL-divergence term `KL(q || p_KB)` pulls decoded states toward stored facts. |
| Crystals + embedding recall | Episodic prior `p(s | context)`. Same role as KB but indexed by embedding similarity rather than HRR query. |
| PCN | Local layer-wise prediction-error minimization. Standard PCN result: stacked PCN minimizes a variational free energy bound when layers are Gaussian (Whittington & Bogacz 2017). |
| LSM with RLS | Recurrent posterior regression. RLS minimizes `E[(o - ŝ)^2]` analytically. |
| Tsetlin clauses | Each clause votes on whether the state matches a learned conjunction. Adds a categorical-prior term `log p(s | clauses)` to the joint. |
| FEP rank-1 A | Online generative-model parameter update. `A ← A + α u v^T` where `u, v` are derived from the current prediction error and the routing direction. |
| ToM beliefs | Conditional prior `p(s | other_agent_model)`. Belief tuples form their own HRR-bound subspace. |
| Self-model | Conditional prior `p(s | self_state)`. RAIN's self-facts form their own subspace. |
| Metacog calibration | Modulates `H[q(s)]` per-relation. Equivalent to a learned temperature parameter conditioned on the relation type. |
| EFE decoder | `argmin_π G(π)` over the 7-source candidate set. |
| NSGA-II evolution | Posterior model selection across alternative generative structures `p_1, p_2, ..., p_N`. Pareto-front maintained over fitness axes (NLL, ECE, FLOPs). |
| Champion/challenger | Bayesian model averaging; promotion when posterior odds cross threshold. |
| Sleep / replay | Offline `F` minimization on stored crystals; consolidates episodic priors into KB. |
| Self-verification | `KL(q(s_emitted) || q(s_internal))` consistency check. Triggers re-decode when divergence is high. |

## 4. Why this is a single objective, not a sum of objectives

Each component minimizes a different *slice* of `F`:

- PCN minimizes the layer-wise prediction-error slice.
- HYMN minimizes the next-state posterior slice (during Phase 1; frozen after).
- Tsetlin minimizes the symbolic-prior fit slice.
- FEP rank-1 minimizes the generative-model-parameter slice.
- LSM RLS minimizes the recurrent-posterior slice.
- KB / crystals minimize the long-term-prior fit slice.
- SOFAR routing minimizes the precision-weighted-error slice.
- EFE decoder minimizes the action-selection slice.

The slices share their inputs (the current state `s_t`, the residual streams,
the codebook). Updates from each slice modify those shared inputs and so
propagate to every other slice's `F`-computation indirectly.

This is the standard active-inference picture: many local rules, one global
quantity being minimized. Whether each component is *literally* deriving its
update from `F` or is approximating it under simplifying assumptions is an
empirical question per component.

## 5. Two-phase training in `F`-minimization terms

### Phase 1 — Amortized inference of `q(s | o)` under fixed `p`

Standard supervised pre-training of HYMN minimizes `-log p(o)` via the
negative-log-likelihood bound. In active-inference terms: we are training
HYMN to be the amortized posterior `q(s | o)`. After Phase 1, HYMN's
parameters are fixed.

### Phase 2 — Online update of `p` itself via local rules

With `q` (= HYMN) fixed, online operation updates the generative model `p`
itself: PCN updates the codebook (which is the codomain of `p(o | s)`); FEP
rank-1 updates `A` (the dynamics of `p(s_{t+1} | s_t)`); Tsetlin updates the
symbolic priors `p(s)`; KB writes update the long-term prior `p(s)`. Each
update is a local-rule step toward minimizing `F` for that slice.

This is the formal reason continual learning doesn't catastrophically forget:
the recognition density `q` is frozen, and the generative model `p` is
updated *additively* via local rules that do not overwrite global weights.

## 6. Open theoretical questions

1. **Convergence of multi-slice local-rule minimization to a coherent `F` minimum.**
   When does the composition of local rules actually drive `F` down? Conditions:
   shared-input updates do not conflict; clipping prevents divergence;
   sleep/replay re-consolidates. These are empirical claims to verify.
2. **Is the multi-signal EFE decoder formally `argmin_π G(π)` or an approximation?**
   The 7-source fusion with NSGA-II-tuned weights is at minimum a coordinate-
   wise approximation. Tighter derivation requires expressing the 7 sources
   as alternative likelihoods over the same observation and using a proper
   mixture model.
3. **Tsetlin clause feedback under FEP.** Tsetlin Type-I/II are originally
   derived from a different (game-theoretic) framework. Re-deriving them as
   `F`-minimization steps is an open theoretical exercise.

## 7. Citations

- Friston, K. (2010). The free-energy principle: a unified brain theory?
  Nat Rev Neurosci 11, 127-138.
- Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error
  backpropagation algorithm in a predictive coding network with local Hebbian
  synaptic plasticity. Neural Computation 29(5).
- Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference: The Free
  Energy Principle in Mind, Brain, and Behavior. MIT Press.
- Smith, R., Friston, K., & Whyte, C. (2022). A step-by-step tutorial on
  active inference and its application to empirical data. J. Math. Psychol.
- Pinchetti et al. (2025). Towards Scaling Deep Neural Networks with
  Predictive Coding. arXiv:2510.23323.
- Granmo, O.-C. (2018). The Tsetlin Machine. arXiv:1804.01508.
- Kanerva, P. (2009). Hyperdimensional Computing. Cognitive Computation.
- Plate, T. A. (2003). Holographic Reduced Representations.
