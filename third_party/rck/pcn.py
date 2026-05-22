"""Predictive Coding Network encoder.

Maps an input vector through a hierarchy where each layer predicts the layer
below it. Updates are LOCAL -- a layer only sees prediction error from the
adjacent layers, never a global gradient. No backprop.

Algorithm sketch (Rao & Ballard / Whittington & Bogacz):
  Layer i carries activations x_i and predicts x_{i-1} via W_{i-1}.
  Prediction error: e_i = x_i - W_i.T @ x_{i+1}  (predicted from layer above)
  Inference loop runs M steps:
    dx_i = -e_i + W_i @ e_{i-1}
    x_i  += lr_x * dx_i
  Weight update (local, Hebbian on errors):
    dW_i = lr_w * outer(e_{i-1}, x_i)
  The top layer is projected to bipolar {-1, +1} to produce a hypervector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rck.vsa import binarize


@dataclass
class PCNEncoder:
    """Hierarchical predictive-coding encoder with local updates.

    Args:
        input_dim: width of the input vector (e.g. one-hot vocab size).
        hidden_dims: list of widths for hidden layers, last one feeds HV.
        hv_dim: output hypervector dimensionality.
        infer_steps: inner iterations of inference per input.
        lr_x: state update rate.
        lr_w: weight update rate.
        seed: rng seed.
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    hv_dim: int = 10_000
    infer_steps: int = 8
    lr_x: float = 0.1
    lr_w: float = 0.01
    seed: int = 0
    activation_clip: float = 6.0
    weight_clip: float = 4.0

    _weights: list[np.ndarray] = field(default_factory=list, init=False)
    _projector: np.ndarray = field(default=None, init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        dims = [self.input_dim, *self.hidden_dims]
        for d_below, d_above in zip(dims[:-1], dims[1:]):
            # W_i maps layer i+1 (above) down to layer i (below) prediction.
            scale = 1.0 / np.sqrt(d_above)
            self._weights.append(self._rng.normal(0, scale, (d_above, d_below)).astype(np.float32))
        # Fixed random projector from top hidden to HV space. Not learned.
        scale = 1.0 / np.sqrt(dims[-1])
        self._projector = self._rng.normal(0, scale, (dims[-1], self.hv_dim)).astype(np.float32)

    def _infer(self, x_input: np.ndarray) -> list[np.ndarray]:
        """Run inference to settle layer activations given the input."""
        x = [x_input.astype(np.float32)]
        for d in self.hidden_dims:
            x.append(np.zeros(d, dtype=np.float32))
        # Iterate: each layer reduces its prediction error vs the layer below.
        for _ in range(self.infer_steps):
            for i in range(len(self._weights) - 1, -1, -1):
                # Predict x[i] from x[i+1] via W[i].T
                pred = self._weights[i].T @ x[i + 1]
                err_below = x[i] - pred
                # Optional top-down error from layer above (skipped for top)
                if i + 1 < len(self._weights):
                    pred_above = self._weights[i + 1].T @ x[i + 2]
                    err_above = x[i + 1] - pred_above
                else:
                    err_above = np.zeros_like(x[i + 1])
                # Update x[i+1]: drift toward minimizing both errors.
                # Clipping keeps activations bounded so large one-hot vocab
                # inputs do not blow up the next matmul.
                dx = self._weights[i] @ err_below - err_above
                x[i + 1] = np.clip(
                    x[i + 1] + self.lr_x * dx,
                    -self.activation_clip, self.activation_clip,
                )
        return x

    def encode(self, x_input: np.ndarray, learn: bool = True) -> np.ndarray:
        """Encode an input vector to a bipolar hypervector.

        If learn=True, perform a local Hebbian weight update on the
        prediction errors at every layer.
        """
        x = self._infer(x_input)
        if learn:
            for i in range(len(self._weights)):
                pred = self._weights[i].T @ x[i + 1]
                err = x[i] - pred
                # Local Hebbian: dW[i] ~ outer(x[i+1], err). Weight clipping
                # bounds growth on long ingestion runs.
                self._weights[i] = np.clip(
                    self._weights[i] + self.lr_w * np.outer(x[i + 1], err),
                    -self.weight_clip, self.weight_clip,
                )
        top = x[-1]
        # Project to HV space and binarize.
        hv_real = top @ self._projector
        return binarize(hv_real)

    def encode_real(self, x_input: np.ndarray) -> np.ndarray:
        """Encode without learning, return the real-valued top representation."""
        x = self._infer(x_input)
        return x[-1].copy()
