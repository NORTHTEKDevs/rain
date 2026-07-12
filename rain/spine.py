"""HYMN sequence engine — Python wrapper over the rain-rs Rust hot path.

The Rust core lives at rain-rs/src/hymn.rs; this module is the thin Python
shim that loads the maturin-built extension and exposes a numpy-friendly API.
"""

from __future__ import annotations

import numpy as np

try:
    from rain._rust import PyHymnModel as _RustHymnModel

    _RUST_AVAILABLE = True
except ImportError:
    _RustHymnModel = None
    _RUST_AVAILABLE = False


class HymnModel:
    """HYMN MLP forward. Bipolar i16 in, bipolar i16 out."""

    def __init__(
        self,
        in_dim: int = 10000,
        hidden_dim: int = 4096,
        out_dim: int = 10000,
        dropout: float = 0.0,
        seed: int = 42,
    ) -> None:
        if not _RUST_AVAILABLE:
            raise RuntimeError(
                "rain._rust extension not built. Run `pip install -e .` "
                "(maturin will compile rain-rs)."
            )
        self._inner = _RustHymnModel(in_dim, hidden_dim, out_dim, dropout, seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, state: np.ndarray, input_: np.ndarray) -> np.ndarray:
        """Forward pass. state and input_ are bipolar int16 arrays of shape (in_dim,)."""
        if state.dtype != np.int16:
            state = state.astype(np.int16)
        if input_.dtype != np.int16:
            input_ = input_.astype(np.int16)
        return self._inner.forward(state, input_)

    def weights(self) -> tuple:
        """Return (W1, W2) as numpy float32 arrays.

        W1 shape (in_dim, hidden_dim), W2 shape (hidden_dim, out_dim).
        """
        if not _RUST_AVAILABLE:
            raise RuntimeError("rain._rust not available")
        return self._inner.weights_w1(), self._inner.weights_w2()


def _python_reference_forward(
    W1: np.ndarray, W2: np.ndarray, state: np.ndarray, input_: np.ndarray
) -> np.ndarray:
    """Bit-identical Python reference of rain-rs/src/hymn.rs forward.

    W1: (in_dim, hidden_dim) float32 row-major (same layout as Rust).
    W2: (hidden_dim, out_dim) float32 row-major.
    state, input_: bipolar int16, shape (in_dim,).
    Returns bipolar int16, shape (out_dim,).
    """
    state = state.astype(np.int16)
    input_ = input_.astype(np.int16)
    # Step 1: combined[d] = sign(state[d] + input_[d])
    combined = np.where((state.astype(np.int32) + input_.astype(np.int32)) >= 0, 1, -1).astype(
        np.int16
    )
    # Step 2: hidden_pre[h] = sum_d W1[d,h] * combined[d]  (= combined @ W1)
    hidden_pre = combined.astype(np.float32) @ W1  # shape (hidden_dim,)
    # Step 3: hidden_signed[h] = sign(hidden_pre[h])
    hidden_signed = np.where(hidden_pre >= 0.0, 1, -1).astype(np.int16)
    # Step 4: out_pre[d] = sum_h W2[h,d] * hidden_signed[h]  (= hidden_signed @ W2)
    out_pre = hidden_signed.astype(np.float32) @ W2  # shape (out_dim,)
    # Step 5: out[d] = sign(out_pre[d])
    return np.where(out_pre >= 0.0, 1, -1).astype(np.int16)
