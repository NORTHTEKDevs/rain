# Copyright 2026 Kristian Baer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SOFAR (Signal-Optimized Frequency-Aligned Routing)
"""
Resonant Channel Mapper
=======================

Performs spectral decomposition (SVD) of a transformer's residual-stream
read/write matrices across all layers, then ranks embedding directions by a
"transmission efficiency" score. The resulting :class:`ChannelMap` is the
substrate SOFAR's beam-steering attention and frequency-layered encoder both
depend on.

Analogy: if the residual stream is the ocean, this module finds the SOFAR
channel - the set of directions in which signal travels furthest with
minimum loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class LayerChannels:
    """Ranked list of privileged directions for one transformer layer.

    Attributes:
        layer_idx: Zero-indexed layer position in the model.
        directions: (k, d_model) array; each row is a unit-norm embedding
            direction, sorted by ``scores`` descending.
        scores: (k,) array of transmission-efficiency scores in [0, 1].
        singular_values: (k,) raw singular values from SVD (diagnostic).

    Equality semantics: dataclass ``eq=False`` is set because the default
    field-by-field ``==`` on numpy arrays returns an element-wise array,
    which then raises ``ValueError: truth value of an array with more
    than one element is ambiguous`` when the dataclass tries to combine
    the per-field results. The custom ``__eq__`` below uses
    ``np.array_equal`` so two LayerChannels with identical content
    compare equal in scalar ``bool`` contexts.
    """

    layer_idx: int
    directions: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayerChannels):
            return NotImplemented
        return (
            self.layer_idx == other.layer_idx
            and np.array_equal(self.directions, other.directions)
            and np.array_equal(self.scores, other.scores)
            and np.array_equal(self.singular_values, other.singular_values)
        )

    def __hash__(self) -> int:  # numpy arrays aren't hashable
        return id(self)

    def top_k(self, k: int) -> LayerChannels:
        """Return a new LayerChannels keeping only the top-``k`` directions.

        Negative ``k`` is rejected loudly: pre-R29 ``top_k(-1)`` was
        translated by Python's slice semantics (``directions[:-1]``)
        into "drop the last direction", which silently lost data when
        a caller typoed a flag like ``--top-k=-1``.
        """
        if k < 0:
            raise ValueError(
                f"top_k must be non-negative, got {k}; pass 0 for no "
                "directions or a positive integer for the top-k slice."
            )
        k = min(k, len(self.scores))
        return LayerChannels(
            layer_idx=self.layer_idx,
            directions=self.directions[:k],
            scores=self.scores[:k],
            singular_values=self.singular_values[:k],
        )


@dataclass(eq=False)
class ChannelMap:
    """Full per-layer channel decomposition of a transformer.

    Attributes:
        model_name: HuggingFace model identifier.
        architecture: Family string (e.g. ``gpt2``, ``llama``, ``mistral``).
        num_layers: Number of transformer blocks.
        d_model: Residual-stream dimension.
        layers: Per-layer channel rankings.
        created_at: ISO-8601 UTC creation timestamp.
        scoring_formula: Name/version of the scoring formula used.

    Equality semantics: see :class:`LayerChannels` - ``eq=False`` so the
    custom ``__eq__`` works around numpy's element-wise comparison
    behavior in the layer list.
    """

    model_name: str
    architecture: str
    num_layers: int
    d_model: int
    layers: list[LayerChannels] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    scoring_formula: str = "default/v0"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelMap):
            return NotImplemented
        if (
            self.model_name != other.model_name
            or self.architecture != other.architecture
            or self.num_layers != other.num_layers
            or self.d_model != other.d_model
            or self.created_at != other.created_at
            or self.scoring_formula != other.scoring_formula
        ):
            return False
        if len(self.layers) != len(other.layers):
            return False
        return all(a == b for a, b in zip(self.layers, other.layers))

    def __hash__(self) -> int:  # contains a list; not hashable by content
        return id(self)

    def layer(self, idx: int) -> LayerChannels:
        """Look up the channel ranking for layer ``idx``."""
        return self.layers[idx]

    def save(self, path: str | Path) -> None:
        """Persist to disk as JSON (directions base64-encoded).

        Creates parent directories as needed so a fresh deployment with
        no ``results/`` directory yet does not fail on the first
        ``sofar map`` invocation. Mirrors the parent-creation behaviour
        of :func:`sofar.save_adapter_checkpoint` so customers see the
        same convention across every SOFAR persistence call.
        """
        import base64

        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "created_at": self.created_at,
            "scoring_formula": self.scoring_formula,
            "layers": [],
        }
        for layer in self.layers:
            payload["layers"].append(
                {
                    "layer_idx": layer.layer_idx,
                    "directions_shape": list(layer.directions.shape),
                    "directions_b64": base64.b64encode(
                        layer.directions.astype(np.float32).tobytes()
                    ).decode("ascii"),
                    "scores": layer.scores.astype(np.float32).tolist(),
                    "singular_values": layer.singular_values.astype(
                        np.float32
                    ).tolist(),
                }
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ChannelMap:
        """Load a ChannelMap previously saved with :meth:`save`."""
        import base64

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        layers: list[LayerChannels] = []
        for l in raw["layers"]:
            directions = np.frombuffer(
                base64.b64decode(l["directions_b64"]), dtype=np.float32
            ).reshape(l["directions_shape"]).copy()  # ensure writable
            layers.append(
                LayerChannels(
                    layer_idx=l["layer_idx"],
                    directions=directions,
                    scores=np.array(l["scores"], dtype=np.float32),
                    singular_values=np.array(
                        l["singular_values"], dtype=np.float32
                    ),
                )
            )
        try:
            return cls(
                # Coerce to declared dataclass types so a malformed
                # JSON (wrong types, e.g. ``num_layers="two"``) fails
                # loudly here rather than seeping downstream into
                # arithmetic that produces a cryptic TypeError later.
                model_name=str(raw["model_name"]),
                architecture=str(raw["architecture"]),
                num_layers=int(raw["num_layers"]),
                d_model=int(raw["d_model"]),
                layers=layers,
                created_at=str(raw.get("created_at", "")),
                scoring_formula=str(raw.get("scoring_formula", "default/v0")),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ChannelMap JSON at {path} has malformed fields: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Architecture detection + projection extraction
# ---------------------------------------------------------------------------


_SUPPORTED_FAMILIES = ("gpt2", "llama", "mistral", "qwen2")


def _unwrap_distributed(model: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module`` if ``model`` is a distributed wrapper.

    Handles ``DistributedDataParallel`` / ``DataParallel`` / FSDP wrappers
    that expose the wrapped model via ``.module``. SOFAR mutates attributes
    on the model (``model.sofar_adapters = ...``) and registers hooks on
    its sublayers; if we did this on a DDP wrapper directly, the
    attributes would live on the WRAPPER not the actual transformer, and
    every ``patched.sofar_adapters`` access via the inner model would
    fail. Unwrapping once keeps SOFAR working transparently inside DDP.

    Recurses to handle nested wrappers (rare but legal: DDP-of-FSDP).
    """
    seen: set[int] = set()
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        if id(model) in seen:
            break  # defensive cycle guard
        seen.add(id(model))
        model = model.module
    return model


def _detect_architecture(model: nn.Module) -> str:
    """Return the architecture family string for a HuggingFace model.

    DDP/FSDP-aware: if ``model`` is a distributed wrapper, the family is
    detected against the underlying ``model.module`` rather than the
    wrapper class.
    """
    model = _unwrap_distributed(model)
    cls = type(model).__name__.lower()
    for family in _SUPPORTED_FAMILIES:
        if family in cls:
            return family
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", "").lower() if cfg is not None else ""
    for family in _SUPPORTED_FAMILIES:
        if family in model_type:
            return family
    return "generic"


def _iter_transformer_blocks(model: nn.Module, family: str) -> list[nn.Module]:
    """Return the ordered list of transformer blocks for a given family."""
    model = _unwrap_distributed(model)
    if family == "gpt2":
        return list(model.transformer.h)
    if family in {"llama", "mistral", "qwen2"}:
        return list(model.model.layers)
    base = getattr(model, "model", model)
    for attr in ("layers", "h", "blocks", "decoder"):
        mod = getattr(base, attr, None)
        if mod is not None and hasattr(mod, "__iter__"):
            return list(mod)
    raise RuntimeError(
        f"Cannot locate transformer blocks for architecture '{family}'."
    )


def _is_conv1d(module: nn.Module) -> bool:
    """True if ``module`` is HuggingFace's ``Conv1D`` (GPT-2 uses this).

    We detect by class name rather than import to avoid a hard dependency
    on ``transformers`` at import time.
    """
    return type(module).__name__ == "Conv1D"


def _extract_dense_weight(module: nn.Module) -> np.ndarray:
    """Return ``module.weight`` as a contiguous fp32 numpy array,
    transparently dequantizing common quantization layouts.

    Handles:

    - Standard ``nn.Linear`` / HuggingFace ``Conv1D``: ``weight.float()``
      already returns a float tensor.
    - bitsandbytes ``Linear8bitLt`` (``Int8Params``): ``weight.float()``
      triggers in-place dequantization to fp32.
    - bitsandbytes ``Linear4bit`` (``Params4bit``): the 4-bit packed
      tensor cannot be cast directly; we call ``bnb.functional
      .dequantize_4bit`` via the layer's stored ``quant_state``.
    - GPTQ / auto-gptq layers (``QuantLinear``): packed weight lives in
      ``qweight``; we look for an explicit ``dequantize()`` method or a
      ``recover_weight()`` shim, falling back to a clear error.
    - AWQ layers: similar — try ``.dequantize()`` first, then bail
      with a fix-it message.

    Raises ``RuntimeError`` with the layer's class name when no known
    dequantization path works, so customers know exactly what to
    install or which layer to avoid.
    """
    # Try the layer-level dequantize() shortcut that bnb 0.43+, AWQ,
    # and GPTQ recent versions all expose. If it returns a fp tensor,
    # we're done.
    dequantize = getattr(module, "dequantize", None)
    if callable(dequantize):
        try:
            w = dequantize()
            if isinstance(w, torch.Tensor) and w.is_floating_point():
                return w.detach().cpu().float().numpy().copy()
        except Exception:
            # Fall through to weight-level paths.
            pass

    # GPTQ-style: packed weights in ``qweight`` (and ``qzeros``,
    # ``scales``), no plain ``.weight`` attribute. Try common
    # recover/dequantize attribute names.
    if not hasattr(module, "weight") or module.weight is None:
        for attr in ("recover_weight", "dequantize_weight"):
            recover = getattr(module, attr, None)
            if callable(recover):
                w = recover()
                if isinstance(w, torch.Tensor):
                    return w.detach().cpu().float().numpy().copy()
        raise RuntimeError(
            f"cannot extract dense weight from {type(module).__name__}: "
            "no .weight, no .dequantize(), no .recover_weight(). "
            "If this is a GPTQ / AWQ / bnb-4bit quantized layer, run "
            "``sofar map`` against the un-quantized model first, then "
            "apply the resulting ChannelMap to your quantized model."
        )

    # bitsandbytes Params4bit: packed 4-bit storage with quant_state.
    # Detect by class name to avoid a hard dependency on bnb.
    weight = module.weight
    if type(weight).__name__ == "Params4bit":
        try:
            import bitsandbytes.functional as bnbf  # type: ignore[import-not-found]
            qs = getattr(weight, "quant_state", None)
            if qs is not None:
                w = bnbf.dequantize_4bit(weight.data, qs).float()
                return w.detach().cpu().numpy().copy()
        except ImportError:
            pass
        raise RuntimeError(
            f"4-bit quantized weight (Params4bit) detected in "
            f"{type(module).__name__} but bitsandbytes.functional"
            ".dequantize_4bit is unavailable. Install ``bitsandbytes`` "
            "or run ``sofar map`` against the un-quantized model."
        )

    # Standard path: nn.Linear, Conv1D, bnb Int8Params (which dequantizes
    # via .float()). detach->cpu->float->numpy->copy is the existing
    # contract; .copy() guards against shared storage with the live
    # parameter (mutating the SVD inputs would silently mutate the
    # model otherwise).
    return weight.detach().cpu().float().numpy().copy()


def _w_read(module: nn.Module, d_model: int) -> np.ndarray:
    """Return the weight of a module that *reads* from the residual stream.

    "Reads" means the layer's input is in residual space (``d_model``) and
    its output is elsewhere (e.g. QKV space, MLP hidden space). We return
    the weight shaped as ``(out_features, d_model)`` regardless of whether
    the underlying module is :class:`torch.nn.Linear` or HuggingFace's
    :class:`Conv1D`, so that stacking + SVD gives right singular vectors in
    residual-stream space.

    Quantization-aware: routes through :func:`_extract_dense_weight`,
    which dequantizes bitsandbytes / GPTQ / AWQ layers transparently.
    """
    w = _extract_dense_weight(module)
    if _is_conv1d(module):
        # Conv1D weight is stored as ``(in, out)``; transpose to ``(out, in=d_model)``.
        return w.T
    # nn.Linear weight is already ``(out, in=d_model)``; no transpose needed.
    return w


def _w_write(module: nn.Module, d_model: int) -> np.ndarray:
    """Return the weight of a module that *writes* to the residual stream.

    "Writes" means the layer's output is in residual space (``d_model``).
    We return the weight shaped as ``(in_features, d_model)`` - input as
    rows, residual as columns - matching the orientation :func:`_w_read`
    produces for the reading modules so everything stacks cleanly.

    Quantization-aware: routes through :func:`_extract_dense_weight`.
    """
    w = _extract_dense_weight(module)
    if _is_conv1d(module):
        # Conv1D weight is ``(in, out=d_model)``; already what we want.
        return w
    # nn.Linear weight is ``(out=d_model, in)``; transpose to ``(in, d_model)``.
    return w.T


def _extract_residual_projections(
    block: nn.Module, family: str, d_model: int
) -> np.ndarray:
    """Stack the residual-stream read/write matrices for a single block.

    Returns a ``(rows, d_model)`` matrix concatenating read and write
    projections onto the residual stream. SVD of this matrix surfaces
    directions that are simultaneously read and written heavily - the
    hallmark of a privileged channel.
    """
    matrices: list[np.ndarray] = []

    if family == "gpt2":
        matrices.append(_w_read(block.attn.c_attn, d_model))
        matrices.append(_w_write(block.attn.c_proj, d_model))
        matrices.append(_w_read(block.mlp.c_fc, d_model))
        matrices.append(_w_write(block.mlp.c_proj, d_model))
    elif family in {"llama", "mistral", "qwen2"}:
        attn = block.self_attn
        mlp = block.mlp
        matrices.append(_w_read(attn.q_proj, d_model))
        matrices.append(_w_read(attn.k_proj, d_model))
        matrices.append(_w_read(attn.v_proj, d_model))
        matrices.append(_w_write(attn.o_proj, d_model))
        matrices.append(_w_read(mlp.gate_proj, d_model))
        matrices.append(_w_read(mlp.up_proj, d_model))
        matrices.append(_w_write(mlp.down_proj, d_model))
    else:
        # Generic fallback: we can't tell reads from writes without
        # architecture knowledge, so include any Linear/Conv1D whose in or
        # out dim equals d_model under the read convention. Callers who
        # need correctness should add an explicit family branch above.
        for m in block.modules():
            if isinstance(m, nn.Linear) or _is_conv1d(m):
                w = _extract_dense_weight(m)
                if w.shape[1] == d_model:
                    matrices.append(w)
                elif w.shape[0] == d_model:
                    matrices.append(w.T)
        if not matrices:
            raise RuntimeError(
                "Generic architecture adapter found no projection modules "
                f"with a d_model={d_model} axis. Provide an explicit family "
                f"adapter for {type(block).__name__}."
            )

    return np.concatenate(matrices, axis=0)


# ---------------------------------------------------------------------------
# Channel scoring
# ---------------------------------------------------------------------------


_FORMULA_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # (alpha=variance, beta=entropy, gamma=alignment). The default profile
    # is the seed used to generate the published v0.2.0 GPT-2 benchmarks;
    # alternatives are provided so customers can A/B them cheaply rather
    # than treating "default/v0" as a magic black box.
    "default/v0": (0.55, 0.20, 0.25),
    "variance-only": (1.0, 0.0, 0.0),
    "alignment-only": (0.0, 0.0, 1.0),
    "entropy-only": (0.0, 1.0, 0.0),
    "uniform": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}


def available_scoring_formulas() -> list[str]:
    """Return the list of recognised ``formula`` names for :func:`score_channel`."""
    return list(_FORMULA_WEIGHTS)


def score_channel(
    singular_values: np.ndarray,
    left_singular: np.ndarray,
    right_singular: np.ndarray,
    layer_idx: int,
    prev_left: np.ndarray | None,
    *,
    formula: str = "default/v0",
) -> np.ndarray:
    """Score each SVD direction for transmission efficiency.

    .. warning::
        **STRATEGIC-TODO (research decision):** This default formula is a
        placeholder. The scoring formula defines what a "resonant channel"
        *is* in SOFAR - every downstream module consumes these scores. See
        the research note below for the tradeoffs, then replace the body.

    The scoring function turns a raw SVD of a layer's projection stack into a
    per-direction score in ``[0, 1]`` describing how good that direction is as
    a transmission channel.

    Args:
        singular_values: ``(k,)`` singular values of the layer's projection
            stack, sorted descending.
        left_singular: ``(d_model, k)`` left singular vectors (columns).
            Each column is a candidate embedding direction.
        right_singular: ``(k, d_proj_rows)`` right singular vectors (rows).
        layer_idx: The current layer's zero-indexed position.
        prev_left: ``(d_model, k)`` left singular vectors from the previous
            layer's decomposition, or None at layer 0. Useful for rewarding
            cross-layer alignment (a direction that reads *and* is carried
            forward is doing real work).

    Returns:
        ``(k,)`` array of scores. The current module normalises to ``[0, 1]``
        and sorts descending before returning.

    Research note - tradeoffs to consider:

    - **Variance preservation** (``sigma^2 / sum(sigma^2)``): maximises raw
      energy captured. Tends to reward the top-1 direction heavily and flatten
      the tail.
    - **Spectral entropy** of the singular-value distribution: rewards broad,
      well-conditioned channels over peaked ones. Good for avoiding
      catastrophic single-direction failure.
    - **Cross-layer alignment** with ``prev_left``: a direction that agrees
      with the previous layer's top directions is part of a persistent
      channel (transmission > transient computation). Usually cosine
      similarity of the top-k subspaces.
    - **Condition-number stability**: reject directions with ``sigma_i /
      sigma_1`` below a threshold; they're numerical noise.

    A reasonable composite might weight all four and clamp to ``[0, 1]``.

    Current implementation (v0.2.0):

    Composite of three terms, weighted alpha + beta + gamma; the
    ``formula`` parameter selects the weight profile. The default is

    - ``alpha = 0.55`` — variance preservation (sigma^2 normalised by sum)
    - ``beta  = 0.20`` — spectral entropy (shannon, normalised by log k);
      rewards broad / well-conditioned channels
    - ``gamma = 0.25`` — cross-layer alignment with previous layer's top-k
      directions (cosine of subspaces); rewards persistent channels

    Alternative profiles are exposed via :func:`available_scoring_formulas`
    so customers can A/B test the contribution of each term cheaply rather
    than rebuilding the composite from scratch.
    """
    if formula not in _FORMULA_WEIGHTS:
        raise ValueError(
            f"unknown scoring formula '{formula}'; expected one of "
            f"{available_scoring_formulas()}"
        )
    sv = singular_values.astype(np.float32)
    sv_squared = sv ** 2
    total = float(sv_squared.sum()) + 1e-12

    # Suppress numpy invalid/divide warnings inside the math block: a
    # degenerate weight matrix can hand us inf or NaN singular values,
    # in which case sv^2/total is NaN. We then explicitly nan_to_num at
    # the end so the public return is always finite and inside [0, 1].
    with np.errstate(invalid="ignore", divide="ignore"):
        # Term 1: variance preservation (per-direction). Normalised so the
        # term sums to ~1 across directions; downstream max-norm rescales.
        var_pres = sv_squared / total

        # Term 2: spectral entropy of the WHOLE singular-value distribution
        # broadcast as a scalar to every direction (so directions in a
        # well-conditioned layer all benefit equally).
        p = sv_squared / total
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        entropy_norm = entropy / max(np.log(max(len(sv), 1)), 1e-12)
        entropy_term = np.full_like(var_pres, np.clip(entropy_norm, 0.0, 1.0))

        # Term 3: cross-layer alignment with previous layer's top-k directions.
        if prev_left is not None and prev_left.shape[0] == left_singular.shape[0]:
            k_top = min(8, prev_left.shape[1], left_singular.shape[1])
            # Unit-normalise both sets of column-direction vectors.
            prev_norms = np.linalg.norm(prev_left[:, :k_top], axis=0, keepdims=True)
            prev_norms[prev_norms == 0] = 1.0
            prev_unit = prev_left[:, :k_top] / prev_norms
            curr_norms = np.linalg.norm(left_singular, axis=0, keepdims=True)
            curr_norms[curr_norms == 0] = 1.0
            curr_unit = left_singular / curr_norms
            # alignment[i] = max_j |<prev_unit[:,j], curr_unit[:,i]>|
            alignments = np.abs(prev_unit.T @ curr_unit)  # (k_top, k)
            align_term = alignments.max(axis=0).astype(np.float32)
        else:
            # Layer 0: no previous layer; use neutral mid-value so alignment
            # neither rewards nor punishes top-1.
            align_term = np.full_like(var_pres, 0.5)

        alpha, beta, gamma = _FORMULA_WEIGHTS[formula]
        composite = alpha * var_pres + beta * entropy_term + gamma * align_term

    # Sanitise non-finite values BEFORE clip (np.clip preserves NaN). A
    # degenerate input must never propagate as NaN through the public
    # boundary; downstream consumers expect a finite [0, 1] score per
    # direction. Defense-in-depth: ``_normalise_and_sort`` repeats this
    # sanitisation on its own input so direct callers and the
    # ``map_channels`` pipeline both see clean output.
    composite = np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def _normalise_and_sort(
    directions: np.ndarray,
    scores: np.ndarray,
    singular_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-normalise directions, sort by score descending, clip scores.

    Sanitises non-finite scores AND non-finite directions before any
    downstream consumer sees them. A degenerate weight matrix - whose
    SVD can produce inf or NaN singular values, then NaN through the
    variance/entropy terms, then NaN-divided directions in the
    normalisation step - cannot silently propagate through the patched
    model and surface as NaN logits at inference time.
    """
    # Sanitise directions BEFORE norm computation: an inf row would
    # otherwise produce inf-norm + inf/inf = NaN unit-vector.
    directions = np.nan_to_num(
        directions.astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = directions / norms
    scores = scores.astype(np.float32)
    # Replace NaN with 0 (treat as no-signal direction) and infinities
    # with the unit-interval extremes before max-norm + clip.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    if scores.max() > 0:
        scores = scores / scores.max()
    scores = np.clip(scores, 0.0, 1.0)
    # Singular values are diagnostic-only but we still sanitise them so
    # a saved ChannelMap JSON never carries inf/NaN that would later
    # confuse a downstream analytics consumer.
    singular_values = np.nan_to_num(
        singular_values.astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    order = np.argsort(-scores)
    return directions[order], scores[order], singular_values[order]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_channels(
    model: nn.Module,
    *,
    model_name: str | None = None,
    top_k: int | None = None,
    formula: str = "default/v0",
) -> ChannelMap:
    """Build a :class:`ChannelMap` for a HuggingFace transformer.

    SVD is performed in NumPy on CPU regardless of where the host model
    lives; projection weights are detached and moved to CPU per-layer so
    very large models (Llama-3 70B, Mistral-8x7B) never keep their full
    weight copy resident in GPU memory.

    Args:
        model: A loaded HuggingFace model (``AutoModel`` / ``AutoModelForCausalLM``).
        model_name: Optional label; defaults to the config's ``_name_or_path``.
        top_k: If set, keep only the top-K directions per layer. Must be
            positive when provided.
        formula: Scoring formula profile. See :func:`available_scoring_formulas`
            for valid names. Persisted on the returned :class:`ChannelMap`
            so reloads remember which weights produced the rankings.

    Returns:
        A fully populated ChannelMap.
    """
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when set, got {top_k}")
    if formula not in _FORMULA_WEIGHTS:
        raise ValueError(
            f"unknown scoring formula '{formula}'; expected one of "
            f"{available_scoring_formulas()}"
        )
    # Unwrap DDP/FSDP so config / .eval() live on the actual transformer,
    # not the wrapper. _detect_architecture and _iter_transformer_blocks
    # both unwrap independently for safety.
    model = _unwrap_distributed(model)
    model.eval()
    family = _detect_architecture(model)
    blocks = _iter_transformer_blocks(model, family)

    cfg = getattr(model, "config", None)
    d_model = int(
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or blocks[0].__dict__.get("hidden_size", 0)
    )
    if d_model <= 0:
        raise RuntimeError(
            "Could not determine d_model from model.config; set "
            "config.hidden_size or config.n_embd before mapping."
        )
    name = model_name or getattr(cfg, "_name_or_path", "unknown")

    channel_layers: list[LayerChannels] = []
    prev_left: np.ndarray | None = None

    for layer_idx, block in enumerate(blocks):
        stack = _extract_residual_projections(block, family, d_model)
        # SVD on (rows, d_model); U: (rows, k), S: (k,), Vt: (k, d_model)
        U, S, Vt = np.linalg.svd(stack, full_matrices=False)
        # We want directions in residual-stream space, which live in the
        # columns of V (equivalently rows of Vt). Transpose so columns.
        left = Vt.T.astype(np.float32)  # (d_model, k)
        right = U.T.astype(np.float32)  # (k, rows)

        scores = score_channel(
            singular_values=S.astype(np.float32),
            left_singular=left,
            right_singular=right,
            layer_idx=layer_idx,
            prev_left=prev_left,
            formula=formula,
        )
        # directions as rows (k, d_model) for downstream consumers
        directions = left.T
        directions, scores, singvals = _normalise_and_sort(
            directions, scores, S.astype(np.float32)
        )
        if top_k is not None:
            directions = directions[:top_k]
            scores = scores[:top_k]
            singvals = singvals[:top_k]

        channel_layers.append(
            LayerChannels(
                layer_idx=layer_idx,
                directions=directions,
                scores=scores,
                singular_values=singvals,
            )
        )
        prev_left = left

    return ChannelMap(
        model_name=name,
        architecture=family,
        num_layers=len(blocks),
        d_model=d_model,
        layers=channel_layers,
        scoring_formula=formula,
    )
