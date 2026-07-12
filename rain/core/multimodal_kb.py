"""Multi-modal fact-hypervector encoders for the v2 KB-Attention buffer.

The v2 architecture treats the KB as a (N_facts, D) tensor of fact-hypervectors.
Where those hypervectors COME FROM is decoupled from the model. v5 used
text triples (s, r, o) -> Codebook + bind + bundle. This module adds
encoders for other modalities:

  * IMAGE -> hypervector via deterministic patch-projection
  * AUDIO -> hypervector via deterministic spectrogram-projection
  * NUMERIC TIMESERIES -> hypervector via bind(time-role, value-role)

All encoders produce the SAME shape (D,) bipolar (or near-bipolar) float
vector, so the KB can hold a mixture of modalities -- the model attends
over them uniformly. This is how RAIN does multi-modal without separate
encoder networks per modality (the way LLMs do): the KB is the unified
multi-modal substrate.

Scope of this module: deterministic, non-learned encoders for the
prototype. Production encoders would be light learned projections from
each modality into the same D-dim hypervector space. The architecture
contract here is the (D,) bipolar output; the input modality is
implementation detail behind the encoder.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _seeded_rng(key: bytes, dim: int) -> np.random.Generator:
    """Deterministic RNG keyed by an arbitrary byte string."""
    h = hashlib.sha256(key).digest()
    seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return np.random.default_rng(seed ^ dim)


def encode_image_patches(image: np.ndarray, dim: int, n_patches: int = 16) -> np.ndarray:
    """Encode an image (H, W, C) into a (dim,) bipolar hypervector.

    Algorithm: split into n_patches x n_patches patches, mean-pool each,
    quantize, bind each patch value with a position-role hypervector,
    bundle.
    """
    if image.ndim != 3:
        raise ValueError(f"expected image shape (H, W, C); got {image.shape}")
    H, W, C = image.shape
    ph, pw = max(1, H // n_patches), max(1, W // n_patches)

    rng = _seeded_rng(b"img-patch-role", dim)
    role = (rng.integers(0, 2, size=(n_patches * n_patches, dim)) * 2 - 1).astype(np.float32)

    # Auto-detect threshold from the image range. uint8 images are [0,255];
    # float-normalized ML-standard images are [0,1]. The previous fixed
    # threshold 128.0 produced an all-negative bipolar for float images,
    # making the hypervector identical regardless of image content.
    img_max = float(image.max())
    threshold = 0.5 if img_max <= 1.0 else 128.0

    acc = np.zeros(dim, dtype=np.float32)
    idx = 0
    for i in range(n_patches):
        for j in range(n_patches):
            patch = image[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            mean = float(patch.mean())
            sign = 1.0 if mean >= threshold else -1.0
            acc += role[idx] * sign
            idx += 1
            if idx >= role.shape[0]:
                break
        if idx >= role.shape[0]:
            break

    return np.sign(acc + (acc == 0)).astype(np.int16).astype(np.float32)


def encode_audio_spectrogram(spectrogram: np.ndarray, dim: int) -> np.ndarray:
    """Encode a (n_mels, n_frames) log-mel spectrogram into a (dim,) bipolar hypervector.

    Algorithm: each (mel, frame) cell binds a mel-band role with a sign
    based on the cell's value vs the spectrogram median. Bundle all cells.
    """
    if spectrogram.ndim != 2:
        raise ValueError(f"expected spectrogram (n_mels, n_frames); got {spectrogram.shape}")
    n_mels, n_frames = spectrogram.shape
    rng = _seeded_rng(b"audio-mel-role", dim)
    mel_role = (rng.integers(0, 2, size=(n_mels, dim)) * 2 - 1).astype(np.float32)
    rng_f = _seeded_rng(b"audio-frame-role", dim)
    frame_role = (rng_f.integers(0, 2, size=(n_frames, dim)) * 2 - 1).astype(np.float32)

    median = float(np.median(spectrogram))
    acc = np.zeros(dim, dtype=np.float32)
    for m in range(n_mels):
        for f in range(n_frames):
            sign = 1.0 if spectrogram[m, f] >= median else -1.0
            acc += mel_role[m] * frame_role[f] * sign
    return np.sign(acc + (acc == 0)).astype(np.int16).astype(np.float32)


def encode_numeric_timeseries(values: np.ndarray, dim: int, max_steps: int = 64) -> np.ndarray:
    """Encode a 1-D numeric timeseries into a (dim,) bipolar hypervector.

    Each timestep binds a position-role with a value-role; bundle. Useful
    for time-series facts in the KB ("AAPL close 2026-05-23 = $185.50",
    "patient.heartrate at minute 5 = 78", etc.).
    """
    if values.ndim != 1:
        raise ValueError(f"expected 1-D values; got {values.shape}")
    n = min(len(values), max_steps)
    rng_p = _seeded_rng(b"ts-pos-role", dim)
    rng_v = _seeded_rng(b"ts-val-role", dim)
    pos_role = (rng_p.integers(0, 2, size=(max_steps, dim)) * 2 - 1).astype(np.float32)

    # Quantize values to 16 bins for the value role
    if n == 0:
        return np.zeros(dim, dtype=np.float32)
    v_min, v_max = float(values[:n].min()), float(values[:n].max())
    v_range = max(1e-9, v_max - v_min)
    bins = np.clip(((values[:n] - v_min) / v_range * 15).astype(int), 0, 15)
    val_role = (rng_v.integers(0, 2, size=(16, dim)) * 2 - 1).astype(np.float32)

    acc = np.zeros(dim, dtype=np.float32)
    for t in range(n):
        acc += pos_role[t] * val_role[bins[t]]
    return np.sign(acc + (acc == 0)).astype(np.int16).astype(np.float32)


def encode_modality(mod: str, payload, dim: int, **kwargs) -> np.ndarray:
    """Generic dispatch: ('text', str) | ('image', ndarray) | ('audio', ndarray) | ('ts', ndarray).

    For text we fall back to a single-token Codebook lookup since the
    full s-r-o binding lives in rain.core.relational.
    """
    if mod == "text":
        from rain.core.relational import Codebook

        cb = Codebook(vocab_size=8192, dim=dim, seed=kwargs.get("seed", 0))
        return cb.vector(str(payload)).astype(np.float32)
    if mod == "image":
        return encode_image_patches(payload, dim, n_patches=kwargs.get("n_patches", 16))
    if mod == "audio":
        return encode_audio_spectrogram(payload, dim)
    if mod == "ts":
        return encode_numeric_timeseries(payload, dim, max_steps=kwargs.get("max_steps", 64))
    raise ValueError(f"unknown modality: {mod}")
