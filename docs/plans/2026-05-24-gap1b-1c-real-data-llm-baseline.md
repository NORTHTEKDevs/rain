# Plan: gap-1b/1c — real data + a real LLM baseline (GPU/API session)

**Created:** 2026-05-24. **Status:** NOT STARTED — requires GPU/API (deferred from
the CPU session that produced Findings 1-19 and the RAIN resonator integration).

## Why this is separate

The resonant core is integrated into RAIN (`rain/core/resonator.py`,
`CompositionalReasoner.extract_all`, `evals/tier1_novelty/scan_capacity.py`). The
Hyperion research established, on synthetic SCAN-shaped tasks, that a learned
encoder + exact resonant composition generalizes (incl. unseen depth, output-only)
where transformers/MLPs get ~0%. The remaining claim — "competitive with an LLM on
compositional splits" — needs real data and a real LLM baseline, which need
GPU/API. Do not make that claim until this plan produces the external number.

## Read first (context)

- `~/projects/active/hyperion/design/MILESTONE-2026-05-24-resonance.md` (19 findings, synthesis, "Path to LLM-competitiveness")
- `~/projects/active/hyperion/design/RESONATOR-FINDINGS.md` (esp. 16-19: neuro-symbolic bridge, real-transformer head-to-head, output-only structure + operator learning)
- `~/projects/active/hyperion/experiments/`: `vsa_end2end.py`, `vsa_end2end_recursive.py` (output-only soft→hardened), `vsa_transformer_hybrid.py` (transformer perception + exact composition)
- RAIN: `rain/core/resonator.py`, `rain/cognition/compose.py` (the integrated core)

## Established thesis (do not re-derive)

Induce the SELECTION (token roles, operators) by gradient; keep the OPERATIONS
exact, algebraic, recursive; generalization rides on operations being
content-independent.

## Goal

1. Move off synthetic onto REAL data: actual SCAN (`~/projects/active/hyperion/data/scan/{addprim_jump,length,...}`) and/or COGS. Use the real input token sequences.
2. Hybrid on real data: learned encoder (small transformer, or distil a pretrained LM's parse) → VSA role/filler structure → exact resonant composition (RAIN `rain/core/resonator.py`) → resonant decode. Push toward OUTPUT-ONLY supervision (Findings 18-19: soft differentiable composition, structure/content separated, hardened at inference).
3. Real baseline that measurably fails compositional generalization: (a) from-scratch transformer seq2seq on the split; if GPU/API allows (b) a large pretrained LLM in-context on the held-out compositional split.
4. Head-to-head on the held-out split. Report honestly.

## Discipline (non-negotiable)

- Verification-gated: every claim backed by a run + committed test.
- Keep honest negatives; report partial/failed results plainly.
- Do NOT claim "beats LLMs" without the actual LLM baseline number on the same held-out split, in-session.
- Scope honesty in writeups; novelty = "synthesis for RAIN", never "first".
- INTERNAL working notes — no patent claims; see LICENSE for terms.
- Update the milestone doc + memory as you go.

## First step

Read the two design docs, confirm GPU/API availability and the real SCAN/COGS data
paths, then establish the transformer baseline before building the hybrid.
