"""Liquid State Machine reservoir.

Fixed random sparse recurrent network -- only the readout is trained.
Spectral radius scaled to satisfy the Echo State Property (ESP), guaranteeing
that the reservoir's response to old inputs fades over time.

State update:
    s(t+1) = (1 - leak) * s(t) + leak * tanh(W_in @ u(t) + W_rec @ s(t))

Readout (real-valued):
    y(t)  = W_out @ s(t)

Online ridge regression updates W_out incrementally without storing the
full reservoir history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rck.vsa import binarize


@dataclass
class LiquidStateMachine:
    """Echo State Network with online ridge readout to HV space.

    Args:
        input_dim: width of input (we feed HV from PCN, so this equals hv_dim).
        reservoir_dim: number of recurrent units.
        hv_dim: output hypervector width.
        spectral_radius: target spectral radius of W_rec (<1 for ESP).
        sparsity: fraction of W_rec entries that are nonzero.
        leak: 0 < leak <= 1, smaller = longer memory.
        ridge: regularization for the readout.
        seed: rng seed.
    """

    input_dim: int
    reservoir_dim: int = 512
    hv_dim: int = 10_000
    spectral_radius: float = 0.9
    sparsity: float = 0.1
    leak: float = 0.3
    ridge: float = 1e-3
    seed: int = 0

    _W_in: np.ndarray = field(default=None, init=False)
    _W_rec: np.ndarray = field(default=None, init=False)
    _W_out: np.ndarray = field(default=None, init=False)
    _P: np.ndarray = field(default=None, init=False)
    _state: np.ndarray = field(default=None, init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        scale_in = 1.0 / np.sqrt(self.input_dim)
        self._W_in = self._rng.normal(0, scale_in, (self.reservoir_dim, self.input_dim)).astype(np.float32)
        W = self._rng.normal(0, 1.0, (self.reservoir_dim, self.reservoir_dim)).astype(np.float32)
        mask = (self._rng.random((self.reservoir_dim, self.reservoir_dim)) < self.sparsity).astype(np.float32)
        W = W * mask
        eigvals = np.linalg.eigvals(W)
        rho = float(np.max(np.abs(eigvals)))
        if rho > 0:
            W *= self.spectral_radius / rho
        self._W_rec = W.astype(np.float32)
        # Readout: trained online via Recursive Least Squares (RLS).
        self._W_out = np.zeros((self.hv_dim, self.reservoir_dim), dtype=np.float32)
        # RLS inverse covariance.
        self._P = (1.0 / self.ridge) * np.eye(self.reservoir_dim, dtype=np.float32)
        self._state = np.zeros(self.reservoir_dim, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros(self.reservoir_dim, dtype=np.float32)

    def step_state(self, u: np.ndarray) -> np.ndarray:
        """Advance one tick. Returns the new real-valued reservoir state."""
        u = u.astype(np.float32)
        drive = self._W_in @ u + self._W_rec @ self._state
        self._state = (1.0 - self.leak) * self._state + self.leak * np.tanh(drive)
        return self._state.copy()

    def readout(self, state: np.ndarray | None = None) -> np.ndarray:
        """Project current (or supplied) state through W_out, return bipolar HV."""
        s = self._state if state is None else state
        y = self._W_out @ s
        return binarize(y)

    def step(self, u: np.ndarray) -> np.ndarray:
        """One full tick: update state, return bipolar HV readout."""
        self.step_state(u)
        return self.readout()

    def train_readout(self, target_hv: np.ndarray) -> None:
        """RLS update of W_out so that readout(state) approaches target_hv.

        Target is bipolar; we fit it as a real-valued regression target.
        """
        s = self._state
        Ps = self._P @ s
        denom = 1.0 + s @ Ps
        gain = Ps / denom
        pred = self._W_out @ s
        err = target_hv.astype(np.float32) - pred
        # Outer-product rank-1 update of W_out and P.
        self._W_out += np.outer(err, gain)
        self._P -= np.outer(gain, Ps)
