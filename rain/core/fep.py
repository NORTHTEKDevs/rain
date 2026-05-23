# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/fep.py -- adapted from RCK 4K FHRR to
# bipolar 10K-dim, simplified for v0 RAIN.
"""Free Energy Principle generative-model parameter.

Low-rank A = U @ V.T predicts s_{t+1} from s_t. Rank-1 update via
append-then-prune keeps rank bounded at R. O(D*R) memory and forward time.
"""

from __future__ import annotations

import numpy as np


class LowRankA:
    def __init__(self, D: int, R: int = 128, seed: int = 0) -> None:
        self.D = D
        self.R = R
        rng = np.random.default_rng(seed)
        # Init U, V small-random (float64 to avoid overflow in rank-1 accumulation)
        scale = 1.0 / np.sqrt(D)
        self.U = rng.standard_normal((D, R)) * scale
        self.V = rng.standard_normal((D, R)) * scale

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Compute A @ state = U @ (V.T @ state). Returns shape (D,)."""
        if state.shape != (self.D,):
            raise ValueError(f"state shape {state.shape} != ({self.D},)")
        s = state.astype(np.float64)
        v_proj = self.V.T @ s  # (R,)
        return (self.U @ v_proj).astype(np.float32)  # (D,)

    def update(self, state: np.ndarray, target: np.ndarray, alpha: float = 0.01) -> float:
        """Rank-1 update: append (alpha * residual) as new u; state as new v; prune lowest-norm column.

        Uses a stabilized variant: the new (u, v) pair encodes the residual in the
        latent space of V (v_proj = V.T @ s), keeping the factorization stable.
        Returns squared residual before the update."""
        if state.shape != (self.D,):
            raise ValueError(f"state shape {state.shape} != ({self.D},)")
        if target.shape != (self.D,):
            raise ValueError(f"target shape {target.shape} != ({self.D},)")
        s = state.astype(np.float64)
        pred = self.predict(state).astype(np.float64)
        residual = target.astype(np.float64) - pred
        sq_residual = float(np.dot(residual, residual))

        # Project state into latent space to get the rank-1 update direction.
        # Normalize both new columns so energy per rank-1 step is bounded by alpha^2.
        v_proj = self.V.T @ s  # (R,)
        v_proj_norm = float(np.linalg.norm(v_proj)) + 1e-9
        r_norm = float(np.linalg.norm(residual)) + 1e-9
        new_u = (alpha * residual / r_norm).reshape(-1, 1)  # unit residual, scaled alpha
        new_v_d = (self.V @ (v_proj / v_proj_norm)).reshape(-1, 1)  # unit state projection

        U_aug = np.concatenate([self.U, new_u], axis=1)  # (D, R+1)
        V_aug = np.concatenate([self.V, new_v_d], axis=1)  # (D, R+1)

        # Prune: always keep the new column (index R); drop the weakest of the original R.
        # The new column encodes the current residual and must enter the factorization.
        existing_contributions = np.linalg.norm(U_aug[:, : self.R], axis=0) * np.linalg.norm(
            V_aug[:, : self.R], axis=0
        )
        drop = int(np.argmin(existing_contributions))
        keep_mask = np.ones(self.R + 1, dtype=bool)
        keep_mask[drop] = False
        self.U = U_aug[:, keep_mask]
        self.V = V_aug[:, keep_mask]
        return sq_residual

    def rank(self) -> int:
        return self.U.shape[1]
