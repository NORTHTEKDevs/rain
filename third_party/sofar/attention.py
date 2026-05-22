# Copyright 2026 Kristian Baer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SOFAR (Signal-Optimized Frequency-Aligned Routing)
"""
Beam-Steering Attention
=======================

A LoRA-style adapter that wraps standard multi-head attention and replaces its
uniform broadcast pattern with a directional **beam** whose geometry is pinned
to the top ChannelMap directions. Three operational modes:

- ``FOCUS``:  concentrate attention energy on a narrow region (point-lookup).
- ``SWEEP``:  scan the beam across the context window (search / scan).
- ``TRACK``:  follow a moving cluster of high-salience tokens (continuity).

Implementation strategy (prototype): the adapter projects hidden states
through a low-rank LoRA into the channel-direction subspace, modulates that
projection by a mode-dependent **position envelope**, and writes the result
back into the residual stream via a ``forward_pre_hook`` on each attention
block. This biases queries, keys, and values simultaneously along the channel
axis, equivalent in expectation to directional Q/K rotation but far simpler to
integrate into an arbitrary HuggingFace model.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import torch
from torch import nn

from sofar.encoder import FrequencyLayeredEncoder
from sofar.mapper import ChannelMap, _unwrap_distributed


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class BeamMode(str, Enum):
    """Beam operational mode."""

    FOCUS = "focus"
    SWEEP = "sweep"
    TRACK = "track"


# ---------------------------------------------------------------------------
# Mode selector (STRATEGIC TODO)
# ---------------------------------------------------------------------------


def select_beam_mode(
    hidden_states: torch.Tensor,
    channel_direction: torch.Tensor,
    layer_idx: int,
    num_layers: int,
) -> BeamMode:
    """Pick the beam mode for the current layer given runtime signals.

    .. warning::
        **STRATEGIC-TODO (research decision):** The mode selector is what turns
        SOFAR from "another static sparsity mask" into a dynamic routing
        layer. The default below picks FOCUS for late layers, SWEEP for
        middle layers, and TRACK for very deep layers - a cheap positional
        heuristic. Replace with a principled policy.

    Args:
        hidden_states: ``(B, T, d_model)`` current-layer input activations.
        channel_direction: ``(d_model,)`` the top ChannelMap direction for
            this layer - rows of ``channel_map.layer(idx).directions``.
        layer_idx: Zero-indexed layer position.
        num_layers: Total number of transformer blocks.

    Returns:
        A :class:`BeamMode`.

    Research note - tradeoffs to consider:

    - **Entropy-driven** (measure attention entropy via a one-time probe;
      high -> SWEEP, low -> FOCUS, moderate -> TRACK): attention's own
      uncertainty signals whether it needs a wide search or a narrow lookup.
      Closed-form, no extra params.
    - **Alignment-driven** (project hidden state onto ``channel_direction``;
      high alignment -> TRACK, low -> SWEEP): uses the channel geometry
      directly. Matches the acoustic metaphor.
    - **Query-content gate** (tiny learned MLP on the pooled hidden state ->
      softmax over 3 modes): most expressive, most parameters.
    - **Stage schedule** (early layers SWEEP for discovery, middle TRACK for
      integration, late FOCUS for decoding): no-op at runtime, cheap, but
      cannot adapt to content.
    - **Mixed** (entropy + alignment composite): usually the strongest.

    Remember: mode-switching *per head* gives finer control than per-layer.
    The current function is per-layer for simplicity; extend the return type
    to ``list[BeamMode]`` of length ``n_heads`` when you are ready.
    """
    # ----- default v0.2.0: alignment + depth composite -----
    # Compute mean-pooled hidden direction and its absolute alignment with
    # this layer's top channel direction. High alignment = signal already
    # lives in the channel -> TRACK to follow it. Low alignment + early
    # layer = SWEEP across context to find what matches. Low alignment +
    # late layer = FOCUS, because exploring further is unlikely to pay off.
    if hidden_states.numel() == 0 or channel_direction.numel() == 0:
        # Fallback to depth schedule if either tensor is empty.
        fraction = (layer_idx + 1) / max(num_layers, 1)
        if fraction < 0.4:
            return BeamMode.SWEEP
        if fraction < 0.75:
            return BeamMode.TRACK
        return BeamMode.FOCUS

    with torch.no_grad():
        # Mean-pooling over the WHOLE sequence dilutes the alignment signal
        # at long context: a single in-channel position is averaged with
        # thousands of unaligned filler vectors, and the resulting metric
        # becomes filler statistics rather than signal-presence.
        # Use only the last min(64, T) positions, where the next-token
        # decoding actually happens and where any preserved early-context
        # signal SHOULD have arrived if the channel is doing its job.
        seq_axis = hidden_states.ndim - 2
        seq_len = hidden_states.shape[seq_axis]
        recent_n = min(64, max(1, seq_len))
        # Slice along the sequence axis without assuming positional rank.
        recent = hidden_states.float().narrow(
            seq_axis, seq_len - recent_n, recent_n,
        )
        # Mean over batch + recent-sequence -> (d_model,)
        mean_h = recent.mean(dim=tuple(range(recent.ndim - 1)))
        h_norm = mean_h / (mean_h.norm() + 1e-8)
        c = channel_direction.float()
        c_norm = c / (c.norm() + 1e-8)
        alignment = float(torch.abs(h_norm @ c_norm).item())

    fraction = (layer_idx + 1) / max(num_layers, 1)

    # Calibrated thresholds: alignment > 0.45 means hidden is meaningfully
    # in-channel. Depth threshold of 0.65 marks the transition from
    # exploratory layers (early/middle) to decoding layers (late).
    if alignment > 0.45:
        return BeamMode.TRACK
    if fraction < 0.65:
        return BeamMode.SWEEP
    return BeamMode.FOCUS


# ---------------------------------------------------------------------------
# Beam-steering adapter
# ---------------------------------------------------------------------------


class BeamSteeringAdapter(nn.Module):
    """LoRA-style adapter producing a beam-modulated residual delta.

    Parameters:
        d_model: Residual-stream dimension.
        channel_directions: ``(k, d_model)`` top-k ChannelMap directions for
            the host layer. Stored as a non-trainable buffer.
        rank: LoRA rank. Defaults to 8.
        num_layers: Total layers in the host model (used by the selector).
        layer_idx: This adapter's layer position.

    The adapter produces ``(B, T, d_model)`` residual deltas. ``B_*`` matrices
    are zero-initialised so the adapter is a no-op at start of training.
    """

    def __init__(
        self,
        *,
        d_model: int,
        channel_directions: np.ndarray | torch.Tensor,
        rank: int = 8,
        num_layers: int,
        layer_idx: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.num_layers = num_layers
        self.layer_idx = layer_idx

        # LoRA down/up pair on the residual stream
        self.lora_down = nn.Linear(d_model, rank, bias=False)
        self.lora_up = nn.Linear(rank, d_model, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Learnable beam geometry.
        # ``beam_gate`` is a plain scalar initialised to 0 so the beam term
        # vanishes at initialisation - together with zero-init on
        # ``lora_up``, this guarantees the adapter is a numerical identity
        # at ``forward`` time, matching the LoRA convention and preventing
        # an untrained adapter from degrading baseline evaluations.
        self.log_beam_width = nn.Parameter(torch.tensor(math.log(8.0)))
        self.beam_gate = nn.Parameter(torch.zeros(1))

        # Channel directions (not trainable; they come from the offline SVD).
        # ``.copy()`` guards against non-writable numpy views (e.g. arrays
        # decoded from base64 buffers by :meth:`ChannelMap.load`), which
        # otherwise trigger PyTorch's read-only-tensor warning.
        if isinstance(channel_directions, np.ndarray):
            channel_directions = torch.from_numpy(np.ascontiguousarray(channel_directions))
        self.register_buffer(
            "channel_directions", channel_directions.detach().float()
        )

        # Learnable per-direction mixing weights: pre-v0.4.2 the adapter
        # consumed ONLY ``channel_directions[0]`` despite the
        # ``top_k_directions`` parameter implying it weighted all k. Now
        # the projection / beam-modulation math uses a learnable convex
        # combination of all k directions. Initialised to a one-hot on
        # the top-1 direction so an untrained adapter is bit-identical
        # to the v0.4.1 behaviour - this preserves published benchmark
        # numbers while letting training learn to redistribute weight
        # across the remaining k - 1 directions.
        k = int(self.channel_directions.shape[0])
        init_mix = torch.zeros(k)
        if k > 0:
            init_mix[0] = 1.0
        self.direction_mix = nn.Parameter(init_mix)

        # Eval-time caches initialised here so attribute access never
        # raises AttributeError on a fresh adapter (init starts in
        # train mode by default, so the combined-direction cache is
        # None and the envelope cache is empty until ``.eval()`` is
        # called). Pre-populating here also makes the attributes
        # introspectable by tools that walk module state without ever
        # running a forward pass.
        self._eval_combined_cache: torch.Tensor | None = None
        self._envelope_cache: dict = {}

    # ---- beam envelope helpers ----------------------------------------

    def _envelope(
        self, seq_len: int, mode: BeamMode, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(T,)`` non-negative envelope for the active mode."""
        if seq_len <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # dtype of beam parameters depends on model precision; cast width to
        # float32 for stable envelope math, then downcast on output.
        width = torch.exp(self.log_beam_width.float()).clamp(min=1.0)
        if mode == BeamMode.FOCUS:
            center = float(seq_len - 1)  # lookup end of sequence
            env = torch.exp(-((positions - center) ** 2) / (2 * width * width))
        elif mode == BeamMode.SWEEP:
            env = 0.5 + 0.5 * torch.sin(
                2 * math.pi * positions / max(seq_len, 1)
            )
        else:  # TRACK
            # soft ramp with mild plateau; tracks a moving cluster centred
            # at 2/3 of the sequence, widens with width parameter
            center = 2 * seq_len / 3
            env = torch.sigmoid(-(positions - center) / width)
        # normalise so the envelope sums to sqrt(T) (energy-preserving).
        # guard against degenerate all-zero envelopes (shouldn't happen for
        # any of the three modes, but defends against future extensions)
        denom = env.sum().clamp(min=1e-6)
        env = env / denom * math.sqrt(max(seq_len, 1))
        return env

    def _combined_direction(self) -> torch.Tensor:
        """Return the ``(d_model,)`` learnable mixture of top-k directions.

        ``direction_mix`` is a ``(k,)`` learnable Parameter. At
        initialisation it is one-hot on direction[0], so the combined
        direction equals ``channel_directions[0]`` (matching v0.4.1
        behaviour bit-for-bit). After training the mix can redistribute
        weight across all k directions, finally making the
        ``top_k_directions`` parameter functionally meaningful.

        Eval-time caching strategy (R60 thread-safe): the cache is
        populated ONCE at the ``.eval()`` transition (in :meth:`train`
        when mode=False), not lazily on first forward. This makes the
        cache safe under concurrent forward calls on free-threaded
        Python (PEP 703 / 3.13+ ``--disable-gil``) - lazy population
        could otherwise race two threads writing the same instance
        attribute. Eager population means writes happen on the
        single-threaded ``.eval()`` call site.
        """
        if not self.training:
            cached = self._eval_combined_cache
            if cached is not None:
                return cached
            # Cache miss in eval mode is unexpected (eval() should have
            # populated it). Fall through to a defensive recompute -
            # this branch only fires if a caller hand-mutates
            # ``self.training`` without going through ``.eval()``.
        weights = self.direction_mix.to(self.channel_directions.dtype)
        return (weights.unsqueeze(-1) * self.channel_directions).sum(dim=0)

    def _envelope_cached(
        self, seq_len: int, mode: BeamMode, device: torch.device
    ) -> torch.Tensor:
        """Return ``_envelope(seq_len, mode, device)`` with eval-time
        memoization keyed by ``(seq_len, mode, device, dtype)``.

        At long context (T=32k+) the per-token envelope cost is
        non-trivial: ``torch.arange + exp/sin/sigmoid + sum + sqrt``
        each forward. Memoizing in eval mode reduces autoregressive
        decode to a tensor lookup. Cache key includes device + dtype
        so multi-device serving and dtype changes don't hit stale
        entries. Maximum entries are bounded by the union of seen
        ``(seq_len, mode)`` pairs - typically a handful.
        """
        if self.training:
            return self._envelope(seq_len, mode, device)
        cache: dict = getattr(self, "_envelope_cache", None) or {}
        key = (int(seq_len), mode.value, str(device), str(self.log_beam_width.dtype))
        cached = cache.get(key)
        if cached is not None:
            return cached
        env = self._envelope(seq_len, mode, device).detach()
        # Direct attribute assignment is GIL-atomic on CPython; under
        # free-threaded builds two threads racing here both compute the
        # same deterministic value, so the last writer wins benignly.
        cache[key] = env
        self._envelope_cache = cache
        return env

    def train(self, mode: bool = True) -> "BeamSteeringAdapter":
        """Toggle train mode AND manage the eval-time caches.

        Entering eval mode (``mode=False``) eagerly pre-populates
        ``_eval_combined_cache`` so subsequent forward calls do not
        race to write the cache attribute on free-threaded builds.
        Entering train mode clears all caches so the next eval
        transition rebuilds from current ``direction_mix``.
        """
        # Always invalidate envelope cache on mode change - the envelope
        # depends on ``log_beam_width`` which trains.
        self._envelope_cache = {}
        if mode:
            self._eval_combined_cache = None
        else:
            # Eager population: write happens on the .eval() call site,
            # not in concurrent forward(). Detach so the cache holds a
            # leaf tensor with no gradient graph.
            with torch.no_grad():
                weights = self.direction_mix.to(self.channel_directions.dtype)
                self._eval_combined_cache = (
                    (weights.unsqueeze(-1) * self.channel_directions).sum(dim=0)
                ).detach()
        return super().train(mode)

    def _channel_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project ``hidden`` onto the learnable top-k channel mixture.

        Returns ``(B, T)`` scalar alignments. v0.4.2+ consumes ALL
        ``top_k_directions`` rows via the learnable ``direction_mix``;
        before that only ``channel_directions[0]`` was used.
        """
        return hidden @ self._combined_direction()  # (B, T)

    # ---- forward ------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, mode: BeamMode | None = None
    ) -> torch.Tensor:
        """Produce a residual delta for this layer.

        Args:
            hidden_states: ``(B, T, d_model)`` incoming activations.
            mode: Optional explicit mode; if None, uses :func:`select_beam_mode`.

        Returns:
            ``(B, T, d_model)`` delta. Callers add this to ``hidden_states``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"expected (B, T, d_model), got {tuple(hidden_states.shape)}"
            )
        _, seq_len, _ = hidden_states.shape

        # ----- identity-at-init fast path -------------------------------
        # When the adapter is at LoRA-convention identity (lora_up.weight
        # all zero AND beam_gate ~0), the full delta is provably zero
        # regardless of input. Pre-v0.4.2 we still ran 3 matmuls + envelope
        # math per layer per forward, paying ~60% latency overhead on CPU
        # for "untrained adapter is exact identity" - a guarantee the
        # README explicitly markets. Now eval-mode untrained adapters
        # short-circuit to zero. The check is two cheap reductions and
        # only runs in eval (training pass always exits early via the
        # ``self.training`` gate so gradient flow is unaffected).
        if (
            not self.training
            and float(self.beam_gate.detach().abs().max()) < 1e-12
            and not bool(self.lora_up.weight.detach().any())
        ):
            return torch.zeros_like(hidden_states)

        if mode is None:
            mode = select_beam_mode(
                hidden_states=hidden_states,
                channel_direction=self.channel_directions[0],
                layer_idx=self.layer_idx,
                num_layers=self.num_layers,
            )

        # LoRA contribution: low-rank delta. Zero at initialisation because
        # ``lora_up.weight`` is zero-initialised.
        delta = self.lora_up(self.lora_down(hidden_states))

        # Beam modulation: envelope weights each position, then we pull the
        # delta toward the LEARNABLE COMBINATION of channel directions
        # proportional to alignment. ``beam_gate`` is zero at init so this
        # term vanishes; training can move it freely positive or negative.
        # Use the eval-time-memoized envelope so 32k-token autoregressive
        # decode doesn't repay the ``arange + exp/sin/sigmoid + sum``
        # cost on every step.
        env = self._envelope_cached(seq_len, mode, hidden_states.device)
        alignment = self._channel_projection(hidden_states)  # (B, T)
        modulation = (alignment * env.unsqueeze(0)).unsqueeze(-1)  # (B, T, 1)
        # v0.4.2+: use the learnable mixture of all top-k directions
        # rather than just ``channel_directions[0]``. At init the mix is
        # one-hot on direction[0] so this is bit-identical to v0.4.1.
        combined = self._combined_direction()  # (d_model,)

        # Final delta: LoRA output + gated directional beam bias. Cast
        # back to the host's dtype so ``hidden + delta`` stays precision-
        # consistent when the host model is in bf16 or fp16.
        delta = delta + self.beam_gate * modulation * combined
        return delta.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Patch helper
# ---------------------------------------------------------------------------


def _find_attention_modules(model: nn.Module, family: str) -> list[nn.Module]:
    """Return the list of per-layer attention modules in forward order."""
    if family == "gpt2":
        return [block.attn for block in model.transformer.h]
    if family in {"llama", "mistral", "qwen2"}:
        return [block.self_attn for block in model.model.layers]
    # generic: first Linear-containing submodule named 'attn' / 'self_attn'
    attns: list[nn.Module] = []
    for _name, mod in model.named_modules():
        if mod.__class__.__name__.lower().endswith(("attention", "attn")):
            attns.append(mod)
    return attns


def _find_first_block(model: nn.Module, family: str) -> nn.Module:
    """Return the first transformer block (FrequencyLayeredEncoder hook target).

    Used by :func:`patch` when ``use_frequency_encoder=True`` so the
    encoder pre-processes hidden states before they reach the first
    attention sublayer.
    """
    if family == "gpt2":
        return model.transformer.h[0]
    if family in {"llama", "mistral", "qwen2"}:
        return model.model.layers[0]
    raise RuntimeError(
        f"Cannot locate first transformer block for family '{family}'; "
        "FrequencyLayeredEncoder wiring requires an explicit family adapter."
    )


def _intercept_hidden_states(
    args: tuple,
    kwargs: dict,
    transform,
) -> tuple | None:
    """Apply ``transform`` to the ``(B, T, d)`` tensor in args or kwargs.

    Shared by the beam-steering adapter hook and the FrequencyLayeredEncoder
    hook so both paths (positional vs kwarg-passed hidden_states) are
    exercised by the same code. Returns ``(new_args, new_kwargs)`` for
    PyTorch's ``register_forward_pre_hook(with_kwargs=True)`` contract,
    or ``None`` if no 3-D tensor was found.
    """
    if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 3:
        new_hidden = transform(args[0])
        return ((new_hidden,) + tuple(args[1:]), kwargs)
    for key in ("hidden_states", "x", "input"):
        val = kwargs.get(key)
        if isinstance(val, torch.Tensor) and val.ndim == 3:
            new_kwargs = dict(kwargs)
            new_kwargs[key] = transform(val)
            return (args, new_kwargs)
    return None


class _AdapterHook:
    """Picklable forward_pre_hook for a single :class:`BeamSteeringAdapter`.

    Pre-R40 the hook was a closure created inside ``patch.<locals>.make_hook``,
    which made the patched model unpicklable: any caller that tried to ship
    the model across processes (multiprocessing DataLoader workers, Ray /
    TorchServe replicas, plain ``copy.deepcopy``) hit
    ``PicklingError: Can't pickle local object``. Promoting the hook to a
    module-level callable class makes the model multiprocess-safe.
    """

    __slots__ = ("adapter",)

    def __init__(self, adapter: "BeamSteeringAdapter") -> None:
        self.adapter = adapter

    def _transform(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.adapter(h)

    def __call__(self, _module: nn.Module, args: tuple, kwargs: dict):
        return _intercept_hidden_states(args, kwargs, self._transform)


class _EncoderHook:
    """Picklable forward_pre_hook for :class:`FrequencyLayeredEncoder`.

    Same rationale as :class:`_AdapterHook`: a module-level class is
    picklable, the ``patch.<locals>`` closure was not.
    """

    __slots__ = ("encoder",)

    def __init__(self, encoder: FrequencyLayeredEncoder) -> None:
        self.encoder = encoder

    def __call__(self, _module: nn.Module, args: tuple, kwargs: dict):
        return _intercept_hidden_states(args, kwargs, self.encoder)


def unpatch(model: nn.Module) -> nn.Module:
    """Remove SOFAR adapters and hooks from a previously :func:`patch` -ed model.

    Idempotent: returns ``model`` unchanged if it was never patched. Calling
    this before :func:`patch` is safe and recommended when you want to swap
    ChannelMaps in-place without accumulating hook handles.

    DDP / FSDP / DataParallel-aware: unwraps the distributed wrapper so
    SOFAR attributes are removed from the actual transformer, not the
    wrapper. Returns the original argument so chained API expressions
    (``unpatched = unpatch(model)``) keep working.
    """
    original = model
    model = _unwrap_distributed(model)
    handles = getattr(model, "sofar_hook_handles", [])
    for h in handles:
        try:
            h.remove()
        except Exception:
            # handles may have been removed already; swallow defensively
            pass
    for attr in (
        "sofar_adapters",
        "sofar_hook_handles",
        "sofar_channel_map",
        "sofar_encoder",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    return original


def patch(
    model: nn.Module,
    channel_map: ChannelMap,
    *,
    rank: int = 8,
    top_k_directions: int = 4,
    seed: int | None = None,
    freeze_base: bool = True,
    use_frequency_encoder: bool = False,
    encoder_mid_window: int = 128,
) -> nn.Module:
    """Inject beam-steering adapters into every attention block.

    Registers a ``forward_pre_hook`` on each attention module that adds the
    adapter's beam-modulated delta to the incoming hidden states. The host
    model is returned mutated; the original ``forward`` methods are untouched
    so the model still runs end-to-end.

    Calling :func:`patch` twice on the same model is safe: any previously
    attached SOFAR hooks and adapters are removed first via :func:`unpatch`.

    Args:
        model: A HuggingFace model. Must have been used to build
            ``channel_map`` (architecture mismatch raises).
        channel_map: Output of :func:`sofar.map_channels`.
        rank: LoRA rank per adapter. Must be positive.
        top_k_directions: How many ChannelMap directions to carry on each
            adapter buffer. Must be positive. **Note (v0.4.x):** the
            BeamSteeringAdapter currently consumes only direction[0] in
            its projection / beam-modulation math; the remaining rows
            are allocated for forward compatibility with planned
            per-head selection (v0.5). Set to 1 if you want the
            smallest possible buffer; the published benchmarks use 4.
        seed: If set, torch's RNG is temporarily seeded during adapter
            construction so LoRA weights are deterministic across runs.
            The caller's RNG state is restored on return.
        freeze_base: When ``True`` (default, LoRA convention), the host
            model's parameters are set to ``requires_grad=False`` and only
            the adapter parameters remain trainable. Set to ``False`` if
            you intend to fine-tune base weights alongside the adapter.
        use_frequency_encoder: When ``True``, instantiate
            :class:`sofar.FrequencyLayeredEncoder` and hook it onto the
            first transformer block so hidden states pass through the
            three-band encode + fuse step before reaching attention. The
            encoder is opt-in to keep published baseline benchmarks
            reproducible; toggle on to exercise SOFAR's third mechanism
            end-to-end. Stored as ``model.sofar_encoder`` and trained
            alongside the adapter when ``freeze_base=True``.

            **Identity-at-init caveat:** unlike the LoRA adapter (which
            ships zero-initialised so the patched model is bit-identical
            to the baseline at step 0), the encoder uses random
            ``nn.Linear`` initialisations and modifies hidden states
            from the first forward pass. An untrained
            ``use_frequency_encoder=True`` model will produce different
            logits from the baseline; train the adapter + encoder
            together via :func:`sofar.train_adapter` before drawing any
            comparison.
        encoder_mid_window: Window size for the MID-band averaging pool
            (only used when ``use_frequency_encoder=True``).

    Returns:
        The same model instance, now with :class:`BeamSteeringAdapter`
        modules registered under ``model.sofar_adapters`` and hooks attached.
        When ``use_frequency_encoder=True``, also exposes
        ``model.sofar_encoder``.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if top_k_directions <= 0:
        raise ValueError(f"top_k_directions must be positive, got {top_k_directions}")

    from sofar.mapper import _detect_architecture

    # DDP / FSDP / DataParallel: unwrap so adapter attributes are
    # registered on the underlying transformer rather than the wrapper.
    # The wrapper class doesn't forward attribute access by default, so
    # patching DDP(model) without unwrap would attach adapters to a
    # ghost the inner model can't see. We patch the inner module; the
    # caller continues using their wrapper handle, and DDP's forward
    # delegates to module.forward which now hits our hooks.
    original = model
    model = _unwrap_distributed(model)
    family = _detect_architecture(model)
    if family != channel_map.architecture:
        raise ValueError(
            f"ChannelMap was built for '{channel_map.architecture}' but model "
            f"is '{family}' - regenerate the ChannelMap."
        )
    attns = _find_attention_modules(model, family)
    if len(attns) != channel_map.num_layers:
        raise ValueError(
            f"ChannelMap has {channel_map.num_layers} layers but model has "
            f"{len(attns)} attention blocks."
        )
    # Validate residual-stream dimension before we register adapters; pre-R10
    # a d_model mismatch was deferred to the first forward pass and surfaced
    # as a cryptic "inconsistent tensor size" RuntimeError from inside the
    # hook closure, leaving the user with no clear remediation path.
    cfg = getattr(model, "config", None)
    model_d_model = int(
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or 0
    )
    if model_d_model > 0 and channel_map.d_model != model_d_model:
        raise ValueError(
            f"ChannelMap.d_model={channel_map.d_model} does not match "
            f"model d_model={model_d_model} - regenerate the ChannelMap "
            "against this model."
        )

    # Idempotent: clear any previously attached adapters and hooks.
    unpatch(model)

    # Optionally seed torch's RNG locally so LoRA init is reproducible,
    # then restore the caller's RNG state on exit. Wrapped in try/finally
    # so that any exception thrown while building adapters does NOT leak
    # the caller's RNG state - even if a future refactor adds raise-points
    # after the seed mutation.
    saved_rng = torch.random.get_rng_state() if seed is not None else None
    if seed is not None:
        torch.manual_seed(int(seed))

    adapters = nn.ModuleList()
    hook_handles: list = []

    try:
        # Move adapters onto the host model's device + dtype so the delta
        # is addable to ``hidden_states`` without a cross-device transfer.
        try:
            host_param = next(model.parameters())
            host_device, host_dtype = host_param.device, host_param.dtype
        except StopIteration:
            host_device, host_dtype = torch.device("cpu"), torch.float32

        for layer_idx, attn in enumerate(attns):
            layer_channels = channel_map.layer(layer_idx).top_k(top_k_directions)
            adapter = BeamSteeringAdapter(
                d_model=channel_map.d_model,
                channel_directions=layer_channels.directions,
                rank=rank,
                num_layers=channel_map.num_layers,
                layer_idx=layer_idx,
            ).to(device=host_device, dtype=host_dtype)
            adapters.append(adapter)

            handles = attn.register_forward_pre_hook(
                _AdapterHook(adapter), with_kwargs=True,
            )
            hook_handles.append(handles)

        model.sofar_adapters = adapters
        model.sofar_hook_handles = hook_handles
        model.sofar_channel_map = channel_map

        # Optionally instantiate and wire the FrequencyLayeredEncoder onto
        # the first transformer block so hidden states are pre-processed
        # through LOW/MID/HIGH band encode + fuse before attention. Default
        # is OFF to keep published benchmark numbers reproducible; opt-in
        # via use_frequency_encoder=True to enable SOFAR's third mechanism
        # end-to-end inside the same patch call.
        if use_frequency_encoder:
            encoder = FrequencyLayeredEncoder(
                d_model=channel_map.d_model,
                mid_window=encoder_mid_window,
                use_learned_gate=True,
            ).to(device=host_device, dtype=host_dtype)
            first_block = _find_first_block(model, family)

            enc_handle = first_block.register_forward_pre_hook(
                _EncoderHook(encoder), with_kwargs=True,
            )
            hook_handles.append(enc_handle)
            model.sofar_encoder = encoder

        if freeze_base:
            # LoRA convention: only the adapter is trainable at init.
            for p in model.parameters():
                p.requires_grad_(False)
            for p in adapters.parameters():
                p.requires_grad_(True)
            # Also unfreeze the encoder when present so it trains alongside
            # the adapter in the calibration loop.
            if hasattr(model, "sofar_encoder"):
                for p in model.sofar_encoder.parameters():
                    p.requires_grad_(True)
    finally:
        # Always restore the caller's RNG, even if adapter construction
        # raised partway through. Otherwise downstream randomness in the
        # caller would silently inherit our local seed.
        if saved_rng is not None:
            torch.random.set_rng_state(saved_rng)

    # Return the ORIGINAL argument (possibly the DDP wrapper) so the
    # caller's reference + downstream `.module.sofar_adapters` access
    # both keep working. The wrapper's forward delegates to
    # `wrapper.module.forward` which now fires our hooks.
    return original
