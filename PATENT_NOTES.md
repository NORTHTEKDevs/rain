# RAIN Patent Notes

**CONFIDENTIAL — PATENT PENDING**
**(c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io**

This document captures the patent-pending claims associated with the RAIN
architecture and its component compositions. It is non-binding marketing text;
the controlling instruments are the filed patent applications and any issued
patents that may result.

---

## Six pending novelty claims

### Claim 1 — Vector Symbolic Architecture as primary generative substrate

A generative language model whose runtime state is a fixed-dimensional bipolar
hypervector evolved sequentially via an MLP update rule operating on bipolar
sign-space; in which the codebook, role-filler bindings, role bundling, and
unbinding operations form the primary representational mechanism for
autoregressive token emission, rather than being used as classification, probing,
or symbolic encoding only.

**Prior art reviewed:** arXiv:2509.25045 (VSA probing of LLMs), arXiv:2511.08767
(VSA for Lisp), arXiv:2507.15779 (reservoir computing as LM). None practice
HRR/Plate/Kanerva binding as the autoregressive generative substrate.

### Claim 2 — Multi-signal Expected Free Energy decoder

A generative decoder that produces the next-token distribution by fusing
candidates drawn from at least seven independently-derived sources, including:
amortized posterior prediction over a VSA state, recurrent recall from a Liquid
State Machine, n-gram retrieval from a VSA bigram memory, action selection
minimizing Expected Free Energy under a low-rank generative model, fact recall
from a sharded HRR knowledge base, structural-validity vote from a vectorized
Tsetlin Machine clause population, and embedding-similar recall from an
episodic crystal store; with the fused distribution then modulated by
per-relation Bayesian calibration weights and a multi-window anti-repetition
penalty before top-p sampling.

### Claim 3 — Structural self-model and theory-of-mind as first-class state objects

A language model architecture in which (a) the model's self-state, (b) the
believed states of any number of external agents, and (c) ground truth facts
are maintained as separate but interoperating HRR-bound subspaces of the
runtime state, such that the model can reason about its own knowledge, the
asymmetric beliefs of multiple external agents, and ground truth simultaneously
during a single forward pass, without prompt engineering or external
scratchpad modules.

**Prior art reviewed:** ToM-agent (Yang 2025), TimeToM (Hou 2024), AutoToM
(2025), arXiv:2603.26089 (LLM ToM deficits). All use prompting or external BDI
modules; none bake self-model and ToM into model weights as state objects.

### Claim 4 — Continual learning by structural construction

A neural-symbolic architecture in which post-bootstrap weight updates are
restricted to local-rule learning operations — Hebbian predictive-coding
updates, Recursive Least Squares posterior regression, Tsetlin Type-I/II
clause feedback, and rank-1 generative-model parameter updates — such that no
global gradient descent occurs at runtime, and therefore the architecture is
structurally immune to catastrophic forgetting by construction rather than by
mitigation.

**Prior art reviewed:** CSUR 2025 continual-learning survey,
arXiv:2504.01241 (catastrophic forgetting in LLMs), ICLR 2025 spurious
forgetting. All gradient-based mitigations; none structurally preclude global
weight overwrite.

### Claim 5 — Per-relation Bayesian calibration as decoding-time signal

A method of generating calibrated outputs from a language model in which the
final token probability distribution is re-weighted at decode time by
per-relation epistemic confidence statistics maintained as a running Bayesian
tally with shrinkage, such that tokens whose source-relation has historically
low calibration are damped and tokens whose source-relation has historically
high calibration are amplified, and the model emits an explicit epistemic
class (knows / thinks / guesses / unknown) for each output.

### Claim 6 — Multi-runtime NSGA-II Pareto evolution with Bayesian champion/challenger promotion

A method of online model improvement in which alternative variants of the
generative model's components (MLP update rules, routing adapters, symbolic
clause sets) are proposed via NSGA-II non-dominated sorting and crowding
distance computation distributed across at least three heterogeneous compute
runtimes (e.g., WASM, native Rust, native Go, TypeScript), and promoted from
challenger to champion status by a Bayesian beta-binomial posterior crossing
an unambiguous credible-interval threshold against the current champion's
posterior, with the entire process running as a non-blocking background
service alongside live inference.

**Prior art reviewed:** EOE (arXiv:2509.24436), EvoMoE (arXiv:2505.23830),
MoSE (arXiv:2602.06154), MFGENAS 2025. None combine NSGA-II Pareto with
multi-runtime distributed evaluation and Bayesian champion/challenger
promotion.

---

## Headers required on every source file

```
// CONFIDENTIAL - PATENT PENDING
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
```

(Adjust comment syntax per language; do not omit either line.)

---

## Public-disclosure rules

Until patent applications have been filed and given enough lead time to grant
priority dates:

- **No public commits, no public branches, no public forks.**
- **No public benchmark posts.** Internal-only benchmarking is fine.
- **No marketing material disclosing component composition** beyond what is
  already public via the source repos' own README files.
- **No paper preprints.** Any external publication waits for explicit IP
  clearance.
- **Crystal content** (project notes, lessons, patterns) is shareable as user
  data, not as kernel architecture. Algorithm names, scoring formulas, the
  composition pipeline, and the WASM/Rust core internals ARE proprietary and
  do not leave the machine without authorization.

---

## Filing checklist (pre-public-launch)

- [ ] Provisional patent application drafted covering Claims 1-6
- [ ] Prior-art search refreshed within 60 days of filing
- [ ] Inventor declaration signed
- [ ] PCT filing strategy decided
- [ ] Trademark search on "RAIN" / "Resonant Active Inference Network" cleared
- [ ] Domain (rain.dev or alternative) acquired and parked
- [ ] All source files carry the patent-pending header (verify with audit
      script in `scripts/`)
- [ ] No copies of the source tree outside controlled NORTHTEKDevs systems
- [ ] All collaborators have signed enforceable confidentiality agreements

---

## Contact

For patent or licensing inquiries: info@northtek.io
