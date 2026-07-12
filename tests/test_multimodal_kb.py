"""Tests for multi-modal KB hypervector encoders."""

import numpy as np

from rain.core.multimodal_kb import (
    encode_audio_spectrogram,
    encode_image_patches,
    encode_modality,
    encode_numeric_timeseries,
)


def test_image_encoder_shape_and_bipolarity():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    hv = encode_image_patches(img, dim=256, n_patches=8)
    assert hv.shape == (256,)
    assert set(np.unique(hv).tolist()).issubset({-1.0, 1.0})


def test_audio_encoder_shape_and_bipolarity():
    spec = np.random.default_rng(0).random((40, 50)).astype(np.float32)
    hv = encode_audio_spectrogram(spec, dim=128)
    assert hv.shape == (128,)
    assert set(np.unique(hv).tolist()).issubset({-1.0, 1.0})


def test_ts_encoder_shape_and_bipolarity():
    series = np.linspace(0, 10, 32).astype(np.float32)
    hv = encode_numeric_timeseries(series, dim=64, max_steps=32)
    assert hv.shape == (64,)
    assert set(np.unique(hv).tolist()).issubset({-1.0, 1.0})


def test_different_images_give_different_hvs():
    a = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    b = (np.random.default_rng(2).random((32, 32, 3)) * 255).astype(np.uint8)
    hv_a = encode_image_patches(a, dim=256)
    hv_b = encode_image_patches(b, dim=256)
    # Cosine should be very low (random patches => roughly orthogonal)
    cos = float(hv_a @ hv_b) / 256
    assert abs(cos) < 0.5  # nowhere near identical


def test_same_input_deterministic():
    img = (np.random.default_rng(42).random((32, 32, 3)) * 255).astype(np.uint8)
    hv1 = encode_image_patches(img, dim=128)
    hv2 = encode_image_patches(img, dim=128)
    np.testing.assert_array_equal(hv1, hv2)


def test_float_normalized_images_produce_different_hvs():
    """Regression: previously the threshold was hardcoded 128.0, which
    silently broke for float-normalized images (all-negative bipolar).
    Two different float images must produce two different hypervectors."""
    rng = np.random.default_rng(0)
    a = rng.random((32, 32, 3)).astype(np.float32)  # values in [0, 1)
    b = rng.random((32, 32, 3)).astype(np.float32)
    hv_a = encode_image_patches(a, dim=128)
    hv_b = encode_image_patches(b, dim=128)
    # The hypervectors must DIFFER (the bug produced identical all-negative hvs)
    assert not np.array_equal(
        hv_a, hv_b
    ), "float-normalized image encoder is broken (all-zero / all-same output)"
    # And both must be proper bipolar
    assert set(np.unique(hv_a).tolist()).issubset({-1.0, 1.0})
    assert set(np.unique(hv_b).tolist()).issubset({-1.0, 1.0})


def test_encode_modality_dispatches():
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    spec = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    ts = np.arange(20, dtype=np.float32)

    h_img = encode_modality("image", img, dim=64)
    h_aud = encode_modality("audio", spec, dim=64)
    h_ts = encode_modality("ts", ts, dim=64)
    h_text = encode_modality("text", "hello", dim=64)

    assert h_img.shape == h_aud.shape == h_ts.shape == h_text.shape == (64,)


def test_encode_modality_unknown_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown modality"):
        encode_modality("video", None, dim=64)
