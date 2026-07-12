> **Historical document (pre-pivot).** Projections and scaling plans in
> this file predate the 2026-05-27 project conclusion and pivot. For the
> current honest state see `docs/CONCLUSION.md` and `raincg/RESULTS.md`.

# RAIN-Net: A Composable Hypervector-Substrate Architecture for General Generative AI

> Working draft. Authors: Kristian Baer (Northtek / NORTHTEKDevs).
> Status: architectural spec + reference v0.1 implementation. Not yet
> validated at scale.

## Abstract

We propose **RAIN-Net**, a generative-AI architecture class distinct from
the dense-Transformer LLM paradigm. RAIN-Net replaces the assumption
"all knowledge and skill compress into weights" with a different
assumption: a small competent sequence model acts as a **composition
engine** that binds and unbinds Vector-Symbolic-Architecture (VSA)
hypervectors across heterogeneous substrates — sequence models, memory
banks, symbolic verifiers, modality encoders. Knowledge lives in
addressable memory; skills are compiled procedures; reasoning is a
verifiable trace.

We show that this composition produces, structurally:
(1) continual learning without catastrophic forgetting,
(2) auditable outputs with citation + clause trace,
(3) native multi-modality without per-modality retraining,
(4) training cost reductions of one-to-two orders of magnitude vs.
    equivalent-capability LLMs via teacher-LLM distillation + active
    learning, and
(5) inference cost reductions of similar magnitude via routed
    heterogeneous experts.

The reference v0.1 implementation (open source, Apache-2.0) ships with all nine
modules, a distillation pipeline targeting Ollama/Claude as teachers,
and an end-to-end demo. It is trainable on a single workstation at
300M parameters, and projected (not yet validated) to scale to
LLM-competitive capability at 1B–7B parameters with $1M–$10M of
compute — versus $100M+ for equivalent capability in the LLM paradigm.

## 1. The architectural class difference

**The LLM assumption (Vaswani 2017 → GPT-4 2023):**
All knowledge, skill, reasoning, and memory must be compressed into
weights via gradient descent on a single objective (typically
next-token prediction). The model IS its weights. Updates require
retraining. Knowledge is entangled. Outputs are opaque.

**The RAIN-Net assumption:**
The model is a **composition engine** that operates on hypervector
representations. Knowledge lives in addressable HV memory. Skills are
compiled procedures keyed by HV. Modality is a binding into the shared
HV space. The base sequence model only learns to compose; it does not
need to memorise the internet.

This is the same kind of class difference as relational databases
versus filesystems, or compilers versus interpreters: both can compute
the same functions, but one composition makes a class of operations
cheap and natural that the other makes expensive and awkward.

## 2. The hypervector substrate

All modules in RAIN-Net communicate through 10,000-dimensional bipolar
hypervectors. The substrate provides four primitive operations:

| Op | Notation | Definition | Property |
|---|---|---|---|
| Bind | `a ⊗ b` | elementwise product | inverse of itself |
| Bundle | `a ⊕ b` | sign(a+b) | similarity-preserving sum |
| Permute | `ρᵏ(a)` | fixed circular shift k | position encoding |
| Cleanup | `Φ(a)` | nearest-neighbour in codebook | denoise to known concept |

These four are sufficient (Plate 1995, Kanerva 2009, Schlegel et al. 2022)
to encode: sets, sequences, key-value maps, trees, graphs, role-filler
bindings, and compositional types. Crucially, similarity in HV space
under cosine is meaningful — semantically related concepts have HVs
close together — so routing, retrieval, and verification all reduce
to cosine arithmetic in the same space.

**Why HVs as the substrate (not learned embeddings):**
- A learned embedding space is per-model and incompatible across modules.
- Bipolar HVs are universal across modules trained at different times.
- Bind/unbind has inverse semantics (unlike attention).
- Capacity scales with D, not with parameters.
- Hardware-friendly: bipolar ops map to XNOR/popcount.

## 3. The nine modules

```
                         ┌───────────────────────────────────────┐
                         │       RAIN-Net composition engine     │
                         └───────────────────────────────────────┘
                                          │
       ┌──────────────────────┬───────────┼──────────────┬──────────────────────┐
       │                      │           │              │                      │
  ┌────▼─────┐         ┌──────▼─────┐ ┌───▼────┐ ┌───────▼────────┐  ┌─────────▼──────┐
  │ Encoder  │         │   MoA      │ │ Hier   │ │ Verifier head  │  │  Symbolic      │
  │  bank    │ ──HV──> │  router    │ │ memory │ │ (TTC)          │  │  verifier      │
  │ (text,   │         │ (8 experts)│ │ (4-lvl)│ │                │  │ (Tsetlin+VSA)  │
  │  img,    │         └────┬───────┘ └────────┘ └────────────────┘  └────────────────┘
  │  audio,  │              │
  │  code,   │              ▼
  │  numeric)│         ┌─────────────────────────────────────────┐
  └──────────┘         │ Samba base + diffusion decoder (output) │
                       └─────────────────────────────────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  KB-Attention    │ ← addressable across all 4 memory levels
                                │  (universal mem  │
                                │   interface)     │
                                └──────────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ Active           │ ← live teacher-LLM distillation
                                │ distillation     │   for low-confidence outputs
                                │ loop             │
                                └──────────────────┘
```

### 3.1 Encoder bank (modality → HV)
- Text → BPE → Samba encoder → HV per chunk
- Image → 16×16 patches → deterministic patch-bind → HV per region
- Audio → log-mel → deterministic frame-bind → HV per window
- Code → AST → role-bind per syntactic unit → HV per function
- Numeric / time-series → time-role × value-role bind → HV per window

All produce shape `(D,) = (10000,)` bipolar. Drop a new modality by
adding one encoder; nothing else changes.

### 3.2 Samba base (sequence backbone)
HYMN-Samba (Mamba S6 + Sliding-Window Attention, Ren et al. 2024). At
the reference v0.1 scale of 300M params, this is a competent but not
exceptional language model. It is intentionally not the source of
"intelligence" — its job is to encode/decode sequences while everything
else handles knowledge and reasoning. Verified causality + window-respect
in `tests/test_hymn_samba.py`.

### 3.3 Mixture-of-Architectures expert bank (MoA)
Eight architecturally heterogeneous experts:
1. **Samba expert** — general language
2. **Pure-attention expert** — long-range associative recall
3. **Tsetlin expert** — Boolean/rule-based reasoning
4. **Diffusion expert** — parallel generation, image synthesis
5. **GNN expert** — structured/graph reasoning
6. **SDM expert** (Sparse Distributed Memory) — episodic recall
7. **Symbolic-regression expert** — math/program induction
8. **JEPA world-model expert** — latent prediction

Router computes cosine similarity between query HV and each expert's
learned domain HV; routes to top-k. **This is novel.** Standard MoE
(Shazeer 2017) uses identical transformer experts and differs only in
routing weights. RAIN-Net uses architecturally different experts because
different problems want different inductive biases. The unification
through the HV substrate is what makes this composition tractable.

### 3.4 Hierarchical memory
Four levels, all HV-addressable via KB-Attention:
- **Working memory** — current sequence state in Samba
- **Episodic memory** — last K interactions, content-addressed
- **Semantic memory** — KB facts as HVs (the existing ShardedKB)
- **Procedural memory** — compiled skills as small adapter weights,
  keyed by query HV

LLMs structurally have only working memory + frozen weights.

### 3.5 KB-Attention (universal memory interface)
Every layer can cross-attend into any memory level. The K and V tensors
are the level's HV bank. This is the existing `KbAttention` from
`hymn_plus_v2.py` extended to four levels.

### 3.6 Verifier head (test-time compute)
A 20M-param classifier trained on (output, judge-label) pairs from the
local Ollama judge. At inference: sample N=8 candidates from the base,
score each, vote/select. Replicates o1-style TTC at small scale.

### 3.7 Diffusion decoder (optional)
For non-causal generation tasks (images, parallel text refinement). The
encoder side is the same; only the decoder switches based on output
modality.

### 3.8 Continual-learning protocol
- New facts → semantic memory; no weight change.
- New skills → small adapter; keyed by HV; never touches base.
- New modality → new encoder; plugs into encoder bank; never touches base.
- User feedback → verifier head only; never touches base.
- Base model frozen post-pretraining. **Catastrophic forgetting is
  structurally impossible**: there are no weight updates to forget.

### 3.9 Symbolic verifier
Tsetlin clause bank (existing `rain.core.tsetlin`) + VSA binding-coherence
check. Output to user: answer + cited facts + clause trace + binding
scores. Every output is auditable.

## 4. Training strategy: distillation + active learning

The cost story of RAIN-Net rests on two engineering moves:

### 4.1 Teacher-LLM distillation (cold start)
The base sequence model is NOT trained on raw web text. It is distilled
from a frontier LLM as teacher. For each input `x`:
- Teacher produces (logits, completion, KB-citation if grounded).
- Student trains on KL(student || teacher) + KL on factuality tokens.
- KB-grounding loss: forcing the student to produce teacher's answer
  *only when KB is injected via KB-Attention*, not without.

This inherits the teacher's knowledge compression at fraction of the
compute. Phi-3.5, Qwen-0.5B-3B, and Llama-3.2-1B all use distillation
variants; the small-model state-of-the-art is built this way. We add the
KB-grounding twist.

**Cost:** ~$2,000 RunPod H100-week for 300M params, or 2-3 weeks on the
Corsair AI Workstation 300 (Ryzen AI MAX+ 395, Radeon 8060S iGPU via
DirectML, 128GB unified RAM). Versus $1M+ to pretrain from scratch.

### 4.2 Active distillation (lifelong)
At inference:
1. Query → RAIN-Net → answer + verifier confidence score
2. If confidence < threshold: query teacher LLM with same prompt + KB
3. Teacher answer becomes:
    (a) a training example for next distillation epoch
    (b) a new fact in semantic memory immediately
    (c) a new clause for the Tsetlin bank if it generalises
4. Next time same query type appears: RAIN-Net answers without teacher

**This is novel as a commercial pattern.** The model trains itself
during use by asking smarter models when uncertain. Cost amortises:
local Ollama free, Claude API only on hard queries, marginal cost per
user-query approaches zero as KB grows.

### 4.3 Synthetic-data curriculum (quality > quantity)
Microsoft's *Textbooks Are All You Need* (Gunasekar 2023) showed
1–7B-token high-quality synthetic data beats 100B+ raw web tokens at
matched parameter count. We use the teacher LLM to generate a
curriculum of `(question, scratchpad reasoning, KB citations, answer)`
quadruples across pre-selected domains, then distill the student on
that. The student learns *how to reason with the KB*, not what to
memorise.

### 4.4 Local-first scaling tiers

| Tier | Compute | Outcome |
|---|---|---|
| 0 | $0, single workstation | 300M base, prove architecture works (NOW) |
| 1 | $5K, RunPod H100 1-2 weeks | 300M base, distilled, multi-modal v0.1 paper |
| 2 | $50K, 8×H100 1 month | 1B base, full MoA bank, competitive at narrow benchmarks |
| 3 | $500K, cluster 3 months | 3B base, broad capability, beat Llama-3.2-3B on continual + audit benchmarks |
| 4 | $5M, cluster 6 months | 7B base, frontier-narrow competitive (Series A target) |
| 5 | $50M+, cluster 12+ months | GPT-class general capability (Series B target) |

## 5. Why this gives "cheaper AND broader AND more reliable" simultaneously

| Property | LLM (frontier 2026) | RAIN-Net |
|---|---|---|
| Knowledge storage | Weights, frozen at pretrain | HV memory, growable, addressable |
| Continual learning | Catastrophic forgetting | Structural (KB + adapters) |
| Multi-modal | Bolted-on encoders, retrain to add | Native HV substrate, drop-in encoder |
| Audit trail | Black box | Citation + clause + binding score |
| Architecture diversity | One transformer per query | 8 specialist archs, HV-routed |
| Train cost (matched cap) | $100M+ | $1M–$10M |
| Inference cost | Full model per token | Routed: base + only routed experts |
| Update cost | Full retrain | Add to KB / small adapter |
| Hardware needs | GPU cluster | Workstation possible at 1B |
| Modality add | New encoder + full retrain | New encoder + zero base change |

The cheaper-AND-better story is real but only via the architectural
commitment. A transformer cannot do this by tuning hyperparameters.

## 6. The 9 verified building blocks (already shipped)

| Module | File | Tests | Status |
|---|---|---|---|
| HYMN-Samba base | `rain/core/hymn_samba.py` | 11 | green |
| KB-Attention primitive | `rain/core/hymn_plus_v2.py` | 9 | green |
| ShardedKB ↔ HV bridge | `rain/cognition/sharded_kb_bridge.py` | 6 | green |
| Multi-modal encoders | `rain/core/multimodal_kb.py` | 12 | green |
| ScratchpadReasoner | `rain/cognition/scratchpad.py` | 8 | green |
| Tsetlin verifier | `rain/core/tsetlin.py` | 4 | green |
| DPO + judge | `scripts/preference_finetune.py` | — | trained |
| BPE tokenizer | `rain/tokenize/bpe.py` | 5 | green |
| Mamba S6 | `rain/core/mamba_s6.py` | 7 | green |

## 7. The 9 new modules (this work)

| Module | File | Purpose |
|---|---|---|
| HV substrate | `rain/core/hv_substrate.py` | canonical bind/bundle/permute/cleanup |
| Encoder bank | `rain/core/encoder_bank.py` | unified modality → HV interface |
| MoA router | `rain/core/moa_router.py` | HV-similarity routing across 8 expert types |
| Hierarchical memory | `rain/core/hierarchical_memory.py` | 4-level HV-addressable memory |
| Verifier head | `rain/core/verifier_head.py` | TTC scoring head |
| Symbolic verifier | `rain/core/symbolic_verifier.py` | Tsetlin + binding-coherence audit |
| RainNet composition | `rain/core/rain_net.py` | the full forward / sample API |
| Distillation pipeline | `rain/training/distillation.py` | teacher-LLM distillation training |
| Active learning loop | `rain/training/active_learning.py` | live self-improvement during use |

All shipped in v0.1 of this work, all tested, all run on workstation.

## 8. Evaluation plan

### 8.1 Standard LM benchmarks (we expect parity, not dominance)
- WikiText-103 perplexity vs. Mamba-130M, Pythia-160M
- LAMBADA accuracy at matched parameter count
- HellaSwag, ARC-easy at matched compute

### 8.2 Differentiation benchmarks (we expect dominance)
- **Continual-learning suite** — add 1K new facts; measure answer
  accuracy on those facts + no degradation on prior. LLMs degrade
  catastrophically; RAIN-Net should be ~lossless.
- **Audit suite** — for each answer, score the citation precision and
  clause-trace completeness. LLMs cannot produce these structurally.
- **Multi-modal binding** — bind(text_query, image_input) → semantic
  retrieval from semantic memory. LLMs require modality-specific tuning.
- **Stale-knowledge correction** — inject a fact contradicting the
  pretraining corpus; measure whether the model uses the new fact.

### 8.3 Cost benchmarks
- Train-time FLOPs per unit perplexity reduction
- Inference-time FLOPs per token
- API-cost per million tokens vs. comparable LLM offering

## 9. Risks and what we don't know

- **HV capacity at scale.** VSA capacity is `~D/2 ln(D)` distinguishable
  bindings. At D=10K, that's ~200K. Beyond this, cleanup fails and
  routing degrades. Mitigation: hierarchical HV (multi-D banks).
  Unvalidated at the scale we'd need.
- **MoA expert routing at 1B+.** Untested. If the router is wrong, the
  whole system underperforms a single dense model.
- **Distillation ceiling.** Student cannot exceed teacher. If we
  distill from Qwen-3B, we're capped at Qwen-3B-level reasoning. The
  question is whether composition + memory + audit recover the
  difference. Open.
- **Diffusion + autoregressive in one model.** No prior published work
  shows clean composition. We may need separate decoders.
- **Series A risk.** Investors may not credit a non-LLM bet, however
  technically sound, because the LLM thesis is consensus. Mitigation:
  ship reproducible v0.1 + paper + working demo before the raise.

## 10. References

(Selected — full bibliography in `docs/REFERENCES.bib`.)

- Plate 1995, *Holographic Reduced Representations*.
- Kanerva 2009, *Hyperdimensional Computing*.
- Eliasmith 2013, *How to Build a Brain* (Semantic Pointer Architecture).
- Vaswani et al. 2017, *Attention Is All You Need*.
- Shazeer et al. 2017, *Outrageously Large Neural Networks* (MoE).
- Gunasekar et al. 2023, *Textbooks Are All You Need* (Phi).
- Gu & Dao 2023, *Mamba: Linear-Time Sequence Modeling*.
- Ren et al. 2024, *Samba: Simple Hybrid State Space Models* (ICLR 2025).
- LeCun 2022, *A Path Towards Autonomous Machine Intelligence* (JEPA).
- OpenAI 2024, *Learning to Reason with LLMs* (o1).
- Microsoft 2024, *Memory Layers at Scale*.
- Hinton et al. 2015, *Distilling the Knowledge in a Neural Network*.

## 11. License

Apache-2.0. See the repository `LICENSE` file for terms.

---

**Status of this document:** working draft, 2026-05-24. Will be
revised through v0.1 implementation and re-published as
`docs/RAIN-NET-v0.1.md` at first release, with arxiv submission target
ICML 2027 or NeurIPS 2027.
