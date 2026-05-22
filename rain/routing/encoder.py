# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/sofar/encoder.py — transplanted from FFT
# spectral partition to Walsh-Hadamard partition for bipolar sign-space.
"""Frequency-banded encoder for RAIN's VSA state.

Decomposes a bipolar 10K-dim hypervector into LOW / MID / HIGH band
projections via Walsh-Hadamard partition. Matches FEP hierarchical
generative model layers:
  - LOW = paragraph-level / slow context drift
  - MID = sentence-level role bindings
  - HIGH = token-level fillers

The original SOFAR uses FFT spectral partition on real-valued transformer
residual streams. For bipolar VSA we substitute Walsh-Hadamard which is
the natural spectral basis for sign-valued signals.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _wht(x: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard transform (sequency order via iterative Hadamard).

    Length must be a power of 2. O(N log N) via butterfly. Returns a fresh
    array (does not modify input)."""
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError(f"length must be power of 2, got {n}")
    y = x.astype(np.float32).copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = y[j]
                b = y[j + h]
                y[j] = a + b
                y[j + h] = a - b
        h *= 2
    return y


def _iwht(x: np.ndarray) -> np.ndarray:
    """Inverse Walsh-Hadamard transform = WHT / N (WHT is self-inverse up to scale)."""
    n = len(x)
    return _wht(x) / n


@dataclass
class FrequencyBands:
    """Result of frequency-banded decomposition."""
    s_L: np.ndarray  # low band, shape (D,)
    s_M: np.ndarray  # mid band, shape (D,)
    s_H: np.ndarray  # high band, shape (D,)
    D: int


def encode(state: np.ndarray) -> FrequencyBands:
    """Decompose a bipolar (or numeric) state into LOW/MID/HIGH bands.

    Args:
        state: shape (D,). Any numeric dtype; cast to float32 internally.
            For bipolar (i16) input, the round-trip s = s_L + s_M + s_H
            recovers the original state within +/- 1 element-wise.

    Returns:
        FrequencyBands with three float32 arrays of shape (D,).
    """
    D = len(state)
    # Pad to next power of 2
    N = _next_pow2(D)
    padded = np.zeros(N, dtype=np.float32)
    padded[:D] = state.astype(np.float32)

    spectrum = _wht(padded)

    # Partition spectrum into three equal-size bands
    third = N // 3
    rem = N - 3 * third
    # Distribute remainder: LOW gets first `third + r1`, MID middle `third + r2`,
    # HIGH last `third`. Spread r1 + r2 = rem evenly.
    low_end = third + (rem // 2)
    mid_end = low_end + third + (rem - rem // 2)
    # high spans [mid_end, N)

    def _band_spectrum(lo: int, hi: int) -> np.ndarray:
        b = np.zeros_like(spectrum)
        b[lo:hi] = spectrum[lo:hi]
        return b

    s_L_padded = _iwht(_band_spectrum(0, low_end))
    s_M_padded = _iwht(_band_spectrum(low_end, mid_end))
    s_H_padded = _iwht(_band_spectrum(mid_end, N))

    return FrequencyBands(
        s_L=s_L_padded[:D].astype(np.float32),
        s_M=s_M_padded[:D].astype(np.float32),
        s_H=s_H_padded[:D].astype(np.float32),
        D=D,
    )


def reconstruct(bands: FrequencyBands) -> np.ndarray:
    """Inverse: s = s_L + s_M + s_H. Returns float32 (caller may quantize)."""
    return bands.s_L + bands.s_M + bands.s_H
