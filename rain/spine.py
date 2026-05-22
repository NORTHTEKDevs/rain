# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
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
