# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/sofar/attention.py -- transplanted from
# transformer attention adapter to VSA-state beam steering.
"""Beam-steering adapter for RAIN routing.

Applies FOCUS/SWEEP/TRACK envelopes along the routing directions from
rain.routing.mapper.RoutingMapper. LoRA-style low-rank adapter with
identity-at-init guarantee (lora_up=0, beam_gate=0).

Energy-normalization keeps total signal power constant across envelope modes
so switching between FOCUS/SWEEP/TRACK doesn't dilute or amplify the state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from rain.routing.mapper import RoutingDirections


class BeamMode(enum.Enum):
    FOCUS = "focus"   # Gaussian -- attend to one direction
    SWEEP = "sweep"   # sinusoidal -- scan across directions
    TRACK = "track"   # sigmoidal -- follow a chain


@dataclass
class BeamConfig:
    mode: BeamMode
    # Center of the envelope along the routing-direction axis (index 0..k-1).
    center: float = 0.0
    # FOCUS: Gaussian std-dev.   SWEEP: wavelength.   TRACK: sigmoid steepness.
    width: float = 1.0


def _envelope(mode: BeamMode, k: int, center: float, width: float) -> np.ndarray:
    """Build the raw envelope along the k routing-direction axis."""
    alpha = np.arange(k, dtype=np.float32)
    if mode is BeamMode.FOCUS:
        # Gaussian
        env = np.exp(-((alpha - center) ** 2) / (2.0 * max(width, 1e-6) ** 2))
    elif mode is BeamMode.SWEEP:
        # Sinusoidal (positive half cycle from |alpha-center|/width)
        env = np.sin(np.pi * (alpha - center) / max(width, 1e-6))
        env = np.abs(env)
    elif mode is BeamMode.TRACK:
        # Sigmoid: 0 -> 1 transition centered at `center`, steepness `width`
        env = 1.0 / (1.0 + np.exp(-(alpha - center) / max(width, 1e-6)))
    else:
        raise ValueError(f"unknown mode {mode}")
    return env.astype(np.float32)


def _energy_normalize(env: np.ndarray) -> np.ndarray:
    """Scale envelope so its L2 norm = sqrt(k) (matches SOFAR's energy convention)."""
    norm = float(np.linalg.norm(env))
    if norm < 1e-9:
        return np.zeros_like(env)
    return env * (np.sqrt(len(env)) / norm)


class BeamSteeringAdapter:
    """LoRA-style adapter projecting the state onto routing directions, applying
    an envelope, and returning the routed state. Identity-at-init."""

    def __init__(self, k: int, D: int, lora_rank: int = 4, seed: int = 0) -> None:
        self.k = k
        self.D = D
        self.lora_rank = lora_rank
        # LoRA factors operating in the routing-direction space (k x rank, rank x k)
        rng = np.random.default_rng(seed)
        self.lora_down = rng.standard_normal((k, lora_rank)).astype(np.float32) * (1.0 / np.sqrt(k))
        # lora_up initialized to ZERO -- identity-at-init guarantee
        self.lora_up = np.zeros((lora_rank, k), dtype=np.float32)
        # Scalar beam gate: also zero at init
        self.beam_gate = np.float32(0.0)

    def apply(
        self,
        state: np.ndarray,
        directions: RoutingDirections,
        config: BeamConfig,
    ) -> np.ndarray:
        """Apply beam steering.

        Args:
            state: (D,) float32 (or castable).
            directions: from RoutingMapper.get(...). V_T shape (k, D).
            config: BeamConfig with mode + center + width.

        Returns:
            routed state, shape (D,) float32.
        """
        if directions.D != self.D:
            raise ValueError(f"directions.D={directions.D} != adapter D {self.D}")
        if directions.k != self.k:
            raise ValueError(f"directions.k={directions.k} != adapter k {self.k}")

        s = state.astype(np.float32)
        # Project state onto routing directions: shape (k,)
        projections = directions.V_T @ s

        # Build + normalize envelope
        env = _envelope(config.mode, self.k, config.center, config.width)
        env_norm = _energy_normalize(env)

        # Weighted projections
        weighted = projections * env_norm

        # LoRA-amplified weighting (zero at init -- no effect)
        if self.beam_gate != 0.0:
            # lora_down: (k, rank), weighted: (k,) -> intermediate: (rank,)
            # lora_up: (rank, k), intermediate: (rank,) -> delta: (k,)
            intermediate = self.lora_down.T @ weighted
            lora_delta = self.lora_up.T @ intermediate
            weighted = weighted + self.beam_gate * lora_delta

        # Re-project back to D-dim
        routed_contribution = directions.V_T.T @ weighted

        # Identity-at-init: when beam_gate=0 AND lora_up=0, output should equal input
        # only if env is uniform AND k = D. In general the projection-and-reconstruction
        # gives a low-rank approximation of the state. To preserve identity-at-init in
        # the strict sense, we return state when beam_gate=0 AND lora_up is all-zero:
        if self.beam_gate == 0.0 and not np.any(self.lora_up):
            return s

        return s + routed_contribution - (directions.V_T.T @ projections)
        # ^ this last term cancels the projection part of the original state so we
        # don't double-count; the result is "original state + steered contribution".
