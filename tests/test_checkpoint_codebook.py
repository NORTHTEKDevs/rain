# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Regression tests for the warm-start codebook persistence fix.

Reproduces the bug: training with a warm-started codebook + saving
without persisting the codebook = mismatched codebook at inference.
"""

import numpy as np

from rain.core.relational import Codebook
from rain.train.checkpoint import save_checkpoint, load_checkpoint, load_codebook, HymnCheckpointMetadata


def _meta(in_dim=64, hidden_dim=32, out_dim=64) -> HymnCheckpointMetadata:
    return HymnCheckpointMetadata(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        steps=100, lr=1e-3, seed=42,
        final_loss=0.5, initial_loss=4.0,
    )


def test_load_codebook_returns_none_for_legacy_checkpoint(tmp_path):
    """Checkpoints without the codebook arrays return None (legacy compat)."""
    W1 = np.random.randn(64, 32).astype(np.float32)
    W2 = np.random.randn(32, 64).astype(np.float32)
    path = tmp_path / "legacy.npz"
    save_checkpoint(path, W1, W2, _meta())
    assert load_codebook(path) is None


def test_codebook_roundtrips_through_save_and_load(tmp_path):
    """Saved codebook chars + matrix come back identical."""
    W1 = np.random.randn(64, 32).astype(np.float32)
    W2 = np.random.randn(32, 64).astype(np.float32)
    chars = ["a", "b", "c", "d"]
    matrix = np.random.choice([-1, 1], size=(4, 64)).astype(np.int16)
    path = tmp_path / "with_codebook.npz"
    save_checkpoint(path, W1, W2, _meta(),
                    codebook_chars=chars, codebook_matrix=matrix)
    loaded = load_codebook(path)
    assert loaded is not None
    loaded_chars, loaded_matrix = loaded
    assert loaded_chars == chars
    assert np.array_equal(loaded_matrix, matrix)


def test_legacy_checkpoint_still_loads_via_load_checkpoint(tmp_path):
    """The non-codebook load path (load_checkpoint) is unchanged for
    legacy checkpoints -- they still return (W1, W2, metadata)."""
    W1 = np.ones((64, 32), dtype=np.float32)
    W2 = np.ones((32, 64), dtype=np.float32)
    path = tmp_path / "legacy.npz"
    save_checkpoint(path, W1, W2, _meta())
    W1b, W2b, meta = load_checkpoint(path)
    assert np.array_equal(W1, W1b)
    assert np.array_equal(W2, W2b)
    assert meta.in_dim == 64
