# Copyright 2026 Kristian Baer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SOFAR (Signal-Optimized Frequency-Aligned Routing)
"""
Signal Fidelity Benchmark
=========================

Measures how well information encoded early in the context window survives
to the model's late-context hidden states. Unlike retrieval-based tasks,
this probes the internal representation directly - we inject a known marker
at position 0 and compare the model's final-layer hidden states at the end
of context against a baseline run where the marker is absent.

The core metric is :func:`compute_fidelity_score` - a strategic research
decision left for the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from sofar.benchmarks import SuiteResult


_MARKER_SENTENCE = (
    "Important marker: the conversation topic is azure horizons. "
    "This marker determines the correct answer below. "
)

_FILLER = (
    "Transformers route information uniformly across the context window. "
    "Whales route their calls along the SOFAR waveguide to minimise loss. "
)

_PROBE = "\n\nThe conversation topic established earlier is"


# ---------------------------------------------------------------------------
# Fidelity score (STRATEGIC TODO)
# ---------------------------------------------------------------------------


def compute_fidelity_score(
    hidden_with_marker: np.ndarray,
    hidden_without_marker: np.ndarray,
    *,
    marker_embedding: np.ndarray | None = None,
) -> float:
    """Quantify how much of the early-context marker survives late in the model.

    .. warning::
        **STRATEGIC-TODO (research decision):** This is the headline metric
        SOFAR will be published against. The default implementation is a
        cosine-distance between marker/no-marker end-of-context hiddens,
        which is only an adequate first-pass. Replace with a principled
        fidelity formulation.

    Args:
        hidden_with_marker: ``(seq_len, d_model)`` last-layer hidden states
            from the run that included the marker.
        hidden_without_marker: ``(seq_len, d_model)`` from the control run
            where the marker was replaced by filler of equal length.
        marker_embedding: Optional ``(d_model,)`` embedding of the marker
            token(s). If provided, the metric should reward preservation of
            marker information specifically (not just any difference).

    Returns:
        Scalar in ``[0, 1]``. Higher = more of the early-context signal
        survives to the late hidden states.

    Research note - options:

    - **Cosine distance** between marker and no-marker end hiddens: cheap,
      measures *any* representational divergence but can be gamed (noise
      also diverges).
    - **Mutual-information lower bound** (InfoNCE between marker / no-marker
      hiddens at each position): principled, expensive, requires a
      contrastive head.
    - **Retrieval-weighted cosine** (multiply cosine by the decoder's
      probability of the correct probe answer): combines geometric and
      behavioural signal, which publishes well.
    - **Linear probe accuracy** (train a tiny classifier on the hiddens to
      recover marker identity): gold standard but adds training cost.
    - **Spectral leakage** (compare singular-value spectra of the two
      trajectories): novel and ties the benchmark back to the mapper's
      mathematics - elegant if it works.

    Any choice must return a single scalar in ``[0, 1]`` so the suite can
    aggregate across context lengths.
    """
    # ----- v0.2.0: position-weighted spectral divergence -----
    # We compare the two trajectories at every position with an exponential
    # weight that emphasises late positions (which is where the early-context
    # marker signal SHOULD have arrived if the model preserved it). The score
    # is the weighted mean of cosine *distance* between the two trajectories.
    h_with = np.asarray(hidden_with_marker, dtype=np.float32)
    h_wo = np.asarray(hidden_without_marker, dtype=np.float32)
    if h_with.ndim != 2 or h_wo.ndim != 2:
        raise ValueError(
            f"hiddens must be 2-D (seq, d_model); got "
            f"{h_with.shape} and {h_wo.shape}"
        )
    if h_with.shape[1] != h_wo.shape[1]:
        raise ValueError(
            f"hidden tensors disagree on d_model axis: "
            f"with={h_with.shape[1]} vs without={h_wo.shape[1]}. "
            "The two trajectories must come from the same model."
        )
    seq = min(h_with.shape[0], h_wo.shape[0])
    if seq == 0:
        return 0.0
    h_with = h_with[:seq]
    h_wo = h_wo[:seq]

    # Bail out on degenerate (all-zero) trajectories - cosine is undefined.
    if float(np.linalg.norm(h_with)) < 1e-6 or float(np.linalg.norm(h_wo)) < 1e-6:
        return 0.0

    # Per-position cosine similarity, robust to zero rows.
    norms_w = np.linalg.norm(h_with, axis=1, keepdims=True)
    norms_w[norms_w < 1e-8] = 1.0
    norms_wo = np.linalg.norm(h_wo, axis=1, keepdims=True)
    norms_wo[norms_wo < 1e-8] = 1.0
    cos = (h_with / norms_w * h_wo / norms_wo).sum(axis=1)
    cos = np.clip(cos.astype(np.float32), -1.0, 1.0)

    # Position weights: exponential from late->early. tau = seq/4 so the
    # last quarter of positions carries roughly half the total weight.
    positions = np.arange(seq, dtype=np.float32)
    tau = max(seq / 4.0, 1.0)
    weights = np.exp(-(seq - 1 - positions) / tau)
    weights = weights / (weights.sum() + 1e-12)

    weighted_cos = float((weights * cos).sum())
    # Map cosine in [-1,1] to fidelity in [0,1] where larger divergence
    # (lower cosine) means more marker signal preserved through the layers.
    fidelity = 1.0 - (weighted_cos + 1.0) / 2.0
    return float(np.clip(fidelity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _build_prompt(
    tokens_target: int, with_marker: bool
) -> str:
    """Build a prompt of approx ``tokens_target`` tokens, w/ or w/o marker.

    LEGACY char-space builder kept for backward compatibility with
    ``tests/test_benchmarks.py``. Production runs use
    :func:`_build_prompt_tokens` so the marker / no-marker pair is exactly
    the same length in TOKEN space (not just char space) - which is what
    matters for per-position alignment in :func:`compute_fidelity_score`.
    """
    chars_target = max(tokens_target, 32) * 4
    head = _MARKER_SENTENCE if with_marker else _FILLER[: len(_MARKER_SENTENCE)]
    filler_needed = max(0, chars_target - len(head) - len(_PROBE))
    reps = filler_needed // len(_FILLER) + 1
    body = (_FILLER * reps)[:filler_needed]
    return head + body + _PROBE


def _tile_tokens(seed_tokens: list[int], n_tokens: int) -> list[int]:
    if n_tokens <= 0 or not seed_tokens:
        return []
    reps = n_tokens // len(seed_tokens) + 1
    return (seed_tokens * reps)[:n_tokens]


def _build_prompt_tokens(
    tokens_target: int,
    with_marker: bool,
    tokenizer: Any,
) -> torch.Tensor:
    """Token-space variant of :func:`_build_prompt`.

    The marker / no-marker pair share head and probe TOKEN counts, so a
    per-position alignment in :func:`compute_fidelity_score` lines up the
    same logical positions in both runs. Char-length matching (legacy
    builder) is only an approximation since marker and filler text
    tokenize differently.
    """
    marker_tok = list(tokenizer.encode(_MARKER_SENTENCE, add_special_tokens=False))
    filler_tok = list(tokenizer.encode(_FILLER, add_special_tokens=False))
    probe_tok = list(tokenizer.encode(_PROBE, add_special_tokens=False))
    if not filler_tok:
        filler_tok = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0]

    head_n = len(marker_tok)
    if with_marker:
        head = list(marker_tok)
    else:
        # Same number of tokens drawn from filler so the no-marker run
        # has identical token-position alignment.
        head = _tile_tokens(filler_tok, head_n)

    body_n = max(0, tokens_target - head_n - len(probe_tok))
    body = _tile_tokens(filler_tok, body_n)

    ids = head + body + probe_tok
    if len(ids) > tokens_target:
        ids = ids[:tokens_target]
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def _hidden_states_for(
    model: Any, tokenizer: Any, input_ids: torch.Tensor
) -> np.ndarray:
    """Run a forward pass and return last-layer hidden states as numpy."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, return_dict=True,
    )
    last_hidden = out.hidden_states[-1]  # (1, T, d_model)
    return last_hidden[0].float().cpu().numpy()


def run_fidelity(
    *,
    model: Any,
    tokenizer: Any | None,
    context_lengths: list[int],
    patched: bool,
    model_name: str,
    seed: int,
    samples: int | None = None,  # accepted for API uniformity; fidelity is deterministic
) -> SuiteResult:
    del samples
    from sofar.benchmarks import SuiteResult
    from sofar.benchmarks.needle import _resolve_tokenizer

    del seed  # deterministic prompts
    tok = _resolve_tokenizer(model, tokenizer)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model.eval()

    per_length: dict[int, float] = {}
    for length in context_lengths:
        with_ids = _build_prompt_tokens(length, with_marker=True, tokenizer=tok)
        without_ids = _build_prompt_tokens(length, with_marker=False, tokenizer=tok)
        h_with = _hidden_states_for(model, tok, with_ids)
        h_wo = _hidden_states_for(model, tok, without_ids)
        per_length[length] = compute_fidelity_score(h_with, h_wo)

    return SuiteResult(
        suite="fidelity",
        model_name=model_name,
        patched=patched,
        per_length=per_length,
        notes="End-of-context divergence between marker and no-marker runs.",
    )
