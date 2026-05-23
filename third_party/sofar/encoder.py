# Copyright 2026 Kristian Baer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SOFAR (Signal-Optimized Frequency-Aligned Routing)
"""
Frequency-Layered Encoder
=========================

Encodes input embeddings simultaneously at three semantic "frequencies":

- ``LOW``:  a single compressed global-intent vector for the full context.
- ``MID``:  paragraph-level coherence embeddings from a sliding window.
- ``HIGH``: standard token-level precision embeddings.

The three bands are merged into a single fused embedding by a learned gating
mechanism (:func:`fuse_bands`, a strategic research decision left to the user).

Inspired by how baleen whales carry meaning across multiple simultaneous
frequency bands in the ocean's SOFAR channel: low-frequency calls carry
long-distance intent, mid-frequency carries social coherence, high-frequency
carries precise identification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Band encoders
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class FrequencyBands:
    """Container for the three encoded bands.

    All tensors are shaped ``(batch, seq_len, d_model)`` so they can be fused
    pointwise. The LOW band is broadcast along the sequence axis so every
    token sees the same global intent vector.

    Equality semantics: ``eq=False`` because the default field-by-field
    ``==`` on torch tensors returns an element-wise tensor, which then
    raises ``RuntimeError: Boolean value of Tensor with more than one
    value is ambiguous`` when the dataclass tries to combine
    per-field results into a scalar bool. Custom ``__eq__`` uses
    ``torch.equal`` for tensor-safe comparison.
    """

    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrequencyBands):
            return NotImplemented
        return (
            torch.equal(self.low, other.low)
            and torch.equal(self.mid, other.mid)
            and torch.equal(self.high, other.high)
        )

    def __hash__(self) -> int:  # tensors aren't hashable
        return id(self)


class LowBandEncoder(nn.Module):
    """Compress the full context into one global-intent vector per batch.

    Uses attention-weighted pooling followed by a linear projection, then
    broadcasts along the sequence dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.attention_score = nn.Linear(d_model, 1, bias=False)
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``(B, T, d_model)`` global-intent tensor.

        Args:
            hidden: ``(B, T, d_model)`` token embeddings.
            mask:   Optional ``(B, T)`` attention mask (1 = keep, 0 = pad).
                Rows that are entirely zero fall back to uniform pooling
                instead of emitting NaN from a softmax of all ``-inf``.
        """
        batch, seq_len, _ = hidden.shape
        logits = self.attention_score(hidden).squeeze(-1)  # (B, T)
        if mask is not None:
            if mask.shape != (batch, seq_len):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != hidden (B,T)="
                    f"{(batch, seq_len)}"
                )
            # If a whole row is masked, fall back to "keep everything" for
            # that row instead of producing NaN via softmax(all -inf).
            row_has_any = mask.sum(dim=1, keepdim=True) > 0  # (B, 1)
            safe_mask = torch.where(
                row_has_any.expand_as(mask), mask, torch.ones_like(mask)
            )
            logits = logits.masked_fill(safe_mask == 0, float("-inf"))
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden * weights).sum(dim=1, keepdim=True)  # (B, 1, d)
        intent = self.projection(pooled)  # (B, 1, d)
        return intent.expand(batch, seq_len, self.d_model)


class MidBandEncoder(nn.Module):
    """Sliding-window paragraph-level coherence encoder.

    For each token, averages embeddings inside a window of ``window_size``
    tokens centred on that position, then applies a small projection. This
    captures paragraph-scale coherence without the cost of full attention.
    """

    def __init__(self, d_model: int, window_size: int = 128):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.d_model = d_model
        self.window_size = window_size
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, d_model)`` windowed coherence embeddings."""
        # 1D average pool along the sequence axis with reflect padding
        # to avoid collapsing edge tokens.
        x = hidden.transpose(1, 2)  # (B, d, T)
        half = self.window_size // 2
        x = F.pad(x, (half, self.window_size - half - 1), mode="replicate")
        x = F.avg_pool1d(
            x, kernel_size=self.window_size, stride=1, padding=0
        )
        x = x.transpose(1, 2)  # (B, T, d)
        return self.projection(x)


class HighBandEncoder(nn.Module):
    """Token-level precision band: a thin residual projection on the input."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.projection(hidden)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fuse_bands(
    low: torch.Tensor,
    mid: torch.Tensor,
    high: torch.Tensor,
    gate_params: BandGate | None = None,
) -> torch.Tensor:
    """Merge the three frequency bands into a single fused embedding.

    .. warning::
        **STRATEGIC-TODO (research decision):** When ``gate_params`` is
        ``None`` this function falls back to a plain equal-weighted sum,
        which is only a placeholder. The fusion mechanism is one of SOFAR's
        three novel mechanisms; replace ``BandGate`` (or pass a custom
        gate) with a principled rule. :class:`FrequencyLayeredEncoder`
        instantiates and supplies a default :class:`BandGate` automatically
        unless you pass ``use_learned_gate=False``.

    Args:
        low:  ``(B, T, d_model)`` LOW-band (global-intent) embedding.
        mid:  ``(B, T, d_model)`` MID-band (paragraph-coherence) embedding.
        high: ``(B, T, d_model)`` HIGH-band (token-precision) embedding.
        gate_params: Optional :class:`BandGate` (or any module with the
            same call signature) holding learned gating parameters. If
            ``None``, a simple equal-weighted sum is used.

    Returns:
        ``(B, T, d_model)`` fused embedding.

    Research note - tradeoffs to consider:

    - **Learned per-token gate** (small MLP on ``high`` -> softmax(3)):
      lets the model decide which band to trust at each token. Best quality,
      most parameters (~3 * d_model^2).
    - **Fixed softmax over global weights**: 3 scalar learned parameters;
      cheapest but the weights don't adapt to context.
    - **Attention-style fusion** (query from HIGH, keys from LOW/MID/HIGH):
      each token attends across its three band-views. Most expressive,
      costs one extra small attention op.
    - **Positional schedule** (early tokens weight LOW more, late tokens
      weight HIGH more): closed-form, no learned params - useful baseline.
    - **Entropy-adaptive** (high attention entropy -> weight LOW more,
      acting like a fog signal): novel and matches the oceanic metaphor.

    Whichever formulation you pick, the output must stay shape-preserving so
    it plugs into the host model unmodified.
    """
    if gate_params is not None:
        return gate_params(low, mid, high)
    # ----- default placeholder: equal-weight mean -----
    return (low + mid + high) / 3.0


class BandGate(nn.Module):
    """Learnable per-token gate over the three frequency bands.

    Reference implementation the user can plug into :func:`fuse_bands` via the
    ``gate_params`` argument. Kept separate from the default so the strategic
    TODO remains visible.

    The forward pass produces a ``(B, T, 3)`` softmax and returns the
    weighted combination.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gate = nn.Linear(d_model, 3)

    def forward(
        self, low: torch.Tensor, mid: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        weights = F.softmax(self.gate(high), dim=-1).unsqueeze(-1)
        stacked = torch.stack([low, mid, high], dim=-2)  # (B, T, 3, d)
        return (weights * stacked).sum(dim=-2)


class EntropyAdaptiveGate(nn.Module):
    """Entropy-adaptive band gate (research-grade alternative to BandGate).

    Conditions the band weights on a measured "uncertainty" signal: the
    entropy of a small attention distribution computed from ``high`` over
    its own sequence positions. High entropy (broad attention -> the model
    is uncertain) shifts weight toward LOW (global intent); low entropy
    (concentrated attention -> the model knows what it's looking at) shifts
    weight toward HIGH (token precision). MID gets the residual mass.

    Matches the SOFAR acoustic metaphor: in fog (uncertainty), whales drop
    to lower-frequency carrier signals that travel further; in clear water
    they use higher-frequency, more precise calls.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # A tiny learned probe that produces per-position attention logits.
        self.probe = nn.Linear(d_model, 1, bias=False)
        # Per-band scaling vectors so the gate can shape output dim.
        self.low_scale = nn.Parameter(torch.ones(d_model))
        self.mid_scale = nn.Parameter(torch.ones(d_model))
        self.high_scale = nn.Parameter(torch.ones(d_model))
        # Temperature applied to the entropy-derived weights.
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, low: torch.Tensor, mid: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        # Compute attention entropy from the high band's own positions.
        # logits: (B, T)
        logits = self.probe(high).squeeze(-1)
        attn = F.softmax(logits, dim=-1)
        # Per-batch entropy in [0, log T]
        entropy = -(attn * torch.log(attn.clamp(min=1e-9))).sum(dim=-1)  # (B,)
        seq_len = max(high.shape[1], 1)
        entropy_norm = entropy / max(math.log(seq_len), 1e-6)  # in [0, 1]
        entropy_norm = entropy_norm.clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)

        # Convert entropy into a 3-way mixture. The temperature controls how
        # sharply we shift between bands; sigmoid(log_temperature) keeps it
        # positive but bounded.
        temp = torch.sigmoid(self.log_temperature) * 2.0 + 0.5  # in (0.5, 2.5)
        # Soft mixture: w_low rises with entropy, w_high falls. w_mid is a
        # tent function peaked at moderate entropy (entropy_norm = 0.5) so
        # the mid band is favored when neither extreme dominates - matches
        # the docstring promise of an entropy-conditioned 3-way mixture.
        w_low = entropy_norm * temp
        w_high = (1.0 - entropy_norm) * temp
        w_mid = (1.0 - (2.0 * entropy_norm - 1.0).abs()) * temp
        weights = F.softmax(torch.cat([w_low, w_mid, w_high], dim=-1), dim=-1)
        # weights shape: (B, 1, 3)
        w_low_b, w_mid_b, w_high_b = weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

        return (
            w_low_b * (low * self.low_scale)
            + w_mid_b * (mid * self.mid_scale)
            + w_high_b * (high * self.high_scale)
        )


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------


class FrequencyLayeredEncoder(nn.Module):
    """Encode hidden states at LOW, MID, HIGH bands and fuse them.

    Pluggable as the first layer (or a pre-processing hook) of any HuggingFace
    transformer. Shape-preserving: input ``(B, T, d_model)`` -> output
    ``(B, T, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        mid_window: int = 128,
        use_learned_gate: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.low = LowBandEncoder(d_model)
        self.mid = MidBandEncoder(d_model, window_size=mid_window)
        self.high = HighBandEncoder(d_model)
        # Spec BR-008 calls for "learned weighted fusion"; the default
        # matches that. Pass ``use_learned_gate=False`` to fall back to the
        # plain-mean placeholder (useful for ablation studies).
        self.gate: BandGate | None = BandGate(d_model) if use_learned_gate else None

    def encode(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> FrequencyBands:
        """Run the three band encoders and return them without fusion."""
        return FrequencyBands(
            low=self.low(hidden, mask=mask),
            mid=self.mid(hidden),
            high=self.high(hidden),
        )

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bands = self.encode(hidden, mask=mask)
        return fuse_bands(bands.low, bands.mid, bands.high, gate_params=self.gate)


def encode_bands(
    hidden: torch.Tensor,
    *,
    d_model: int | None = None,
    mid_window: int = 128,
    mask: torch.Tensor | None = None,
) -> FrequencyBands:
    """Convenience functional wrapper for a one-shot encode.

    Builds a throwaway :class:`FrequencyLayeredEncoder` with fresh weights -
    only useful for diagnostics and smoke tests; normal usage should
    instantiate :class:`FrequencyLayeredEncoder` once and reuse it.
    """
    d = d_model or hidden.shape[-1]
    encoder = FrequencyLayeredEncoder(d_model=d, mid_window=mid_window)
    encoder.eval()
    with torch.no_grad():
        return encoder.encode(hidden, mask=mask)
