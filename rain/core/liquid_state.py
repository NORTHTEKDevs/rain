# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/lsm.py -- adapted from 4K FHRR to bipolar
# 10K-dim, simplified for v0 RAIN.
"""Liquid State Machine with Recursive Least Squares readout.

A small reservoir (typical N=512) is fed bipolar hypervector inputs; its
internal state evolves as tanh(W_res @ s + W_in @ x). A learned linear
readout W_out maps reservoir state to predicted next-hypervector. RLS
gives an exact online least-squares update for W_out.
"""

from __future__ import annotations

import numpy as np


class LiquidStateMachine:
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 512,
        output_dim: int | None = None,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        sparsity: float = 0.1,
        rls_forget: float = 0.99,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        rng = np.random.default_rng(seed)

        # Sparse signed reservoir matrix
        mask = rng.random((reservoir_dim, reservoir_dim)) < sparsity
        signs = rng.choice([-1.0, 1.0], size=(reservoir_dim, reservoir_dim))
        W_res = (mask * signs).astype(np.float32)
        # Scale so spectral radius approx matches target. Use power-iter approx.
        # For a sparse signed matrix, the leading eigenvalue scales with sqrt(sparsity * N).
        approx_radius = np.sqrt(sparsity * reservoir_dim)
        if approx_radius > 1e-6:
            W_res *= spectral_radius / approx_radius
        self.W_res = W_res

        # Input projection: dense small-magnitude random
        self.W_in = (
            rng.standard_normal((reservoir_dim, input_dim)).astype(np.float32)
            * input_scale
            / np.sqrt(input_dim)
        )

        # Readout: zero-init
        self.W_out = np.zeros((self.output_dim, reservoir_dim), dtype=np.float32)

        # RLS state
        self.P = np.eye(reservoir_dim, dtype=np.float32) * 100.0  # initial inverse-corr matrix
        self.rls_forget = rls_forget

        # Reservoir state
        self.state = np.zeros(reservoir_dim, dtype=np.float32)

    def step(self, input_: np.ndarray) -> np.ndarray:
        """Advance reservoir state one step on input. Returns new reservoir state."""
        if input_.shape != (self.input_dim,):
            raise ValueError(f"input dim {input_.shape} != ({self.input_dim},)")
        x = input_.astype(np.float32)
        pre = self.W_res @ self.state + self.W_in @ x
        self.state = np.tanh(pre)
        return self.state

    def reset(self) -> None:
        self.state = np.zeros(self.reservoir_dim, dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Return the readout's prediction for the current reservoir state."""
        return self.W_out @ self.state

    def update(self, target: np.ndarray) -> float:
        """One RLS step on the current reservoir state vs target. Returns squared residual."""
        if target.shape != (self.output_dim,):
            raise ValueError(f"target dim {target.shape} != ({self.output_dim},)")
        y_pred = self.predict()
        residual = target.astype(np.float32) - y_pred

        # RLS gain
        Ps = self.P @ self.state
        denom = self.rls_forget + self.state @ Ps
        gain = Ps / denom

        # Update W_out: each row updated independently (vectorized)
        self.W_out += np.outer(residual, gain)

        # Update P
        self.P = (self.P - np.outer(gain, Ps)) / self.rls_forget

        return float(np.dot(residual, residual))
