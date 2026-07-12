import numpy as np
import pytest

from rain.train.checkpoint import (
    FrozenCheckpointError,
    HymnCheckpointMetadata,
    assert_not_frozen,
    freeze_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_save_load_round_trip(tmp_path):
    W1 = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    W2 = np.random.RandomState(1).randn(8, 4).astype(np.float32)
    meta = HymnCheckpointMetadata(
        in_dim=4,
        hidden_dim=8,
        out_dim=4,
        steps=100,
        lr=0.01,
        seed=42,
        final_loss=0.5,
        initial_loss=1.5,
    )
    npz_path, json_path = save_checkpoint(tmp_path / "ckpt", W1, W2, meta)
    assert npz_path.exists() and json_path.exists()
    W1_loaded, W2_loaded, meta_loaded = load_checkpoint(tmp_path / "ckpt")
    assert np.allclose(W1, W1_loaded)
    assert np.allclose(W2, W2_loaded)
    assert meta_loaded.in_dim == 4
    assert meta_loaded.frozen is False


def test_freeze_marks_metadata(tmp_path):
    W1 = np.zeros((2, 2), dtype=np.float32)
    W2 = np.zeros((2, 2), dtype=np.float32)
    meta = HymnCheckpointMetadata(
        in_dim=2,
        hidden_dim=2,
        out_dim=2,
        steps=0,
        lr=0.01,
        seed=0,
        final_loss=None,
        initial_loss=None,
    )
    save_checkpoint(tmp_path / "ckpt", W1, W2, meta)
    frozen_meta = freeze_checkpoint(tmp_path / "ckpt")
    assert frozen_meta.frozen is True
    _, _, reloaded = load_checkpoint(tmp_path / "ckpt")
    assert reloaded.frozen is True


def test_assert_not_frozen_passes_for_unfrozen():
    meta = HymnCheckpointMetadata(
        in_dim=1,
        hidden_dim=1,
        out_dim=1,
        steps=0,
        lr=0.01,
        seed=0,
        final_loss=None,
        initial_loss=None,
        frozen=False,
    )
    assert_not_frozen(meta)  # no exception


def test_assert_not_frozen_raises_for_frozen():
    meta = HymnCheckpointMetadata(
        in_dim=1,
        hidden_dim=1,
        out_dim=1,
        steps=0,
        lr=0.01,
        seed=0,
        final_loss=None,
        initial_loss=None,
        frozen=True,
    )
    with pytest.raises(FrozenCheckpointError):
        assert_not_frozen(meta)
