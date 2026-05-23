# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HYMN checkpoint format -- save + load + frozen-flag enforcement.

The checkpoint pair is (.npz, .json sidecar):
- .npz contains the float32 weight arrays (W1, W2) and training metadata.
- .json sidecar contains human-readable metadata + the frozen flag.

The frozen flag is set after Phase 1 bootstrap pre-training completes. Any
loader that opens a frozen checkpoint MUST NOT mutate the weights -- Phase 2
continual learning operates on other state, never on HYMN weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class HymnCheckpointMetadata:
    in_dim: int
    hidden_dim: int
    out_dim: int
    steps: int
    lr: float
    seed: int
    final_loss: float | None
    initial_loss: float | None
    frozen: bool = False
    schema_version: int = 1
    # New in schema_version >= 2. Optional so older checkpoint sidecars still load.
    loss_type: str = "mse"        # "mse" or "nll" -- which loss the trainer minimized
    context_len: int = 0          # sequence-context size used during training
    carry_steps: int = 0          # RNN-style carry window (0 = per-position training)
    batch_size: int = 1           # per-step batch size used during training


def save_checkpoint(
    path: str | Path,
    W1: np.ndarray,
    W2: np.ndarray,
    metadata: HymnCheckpointMetadata,
    losses: np.ndarray | None = None,
    codebook_chars: list[str] | None = None,
    codebook_matrix: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Save .npz weights + .json sidecar. Returns (npz_path, json_path).

    If `codebook_chars` and `codebook_matrix` are supplied, they're saved
    alongside W1/W2 so inference reproduces the EXACT codebook the model
    was trained against. Critical for warm-started training, where the
    codebook is no longer a function of (seed, vocab_size, dim) alone.
    """
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    arrays = {"W1": W1.astype(np.float32), "W2": W2.astype(np.float32)}
    if losses is not None:
        arrays["losses"] = losses.astype(np.float32)
    if codebook_matrix is not None:
        arrays["codebook_matrix"] = codebook_matrix.astype(np.int16)
    np.savez(npz_path, **arrays)
    meta_dict = asdict(metadata)
    if codebook_chars is not None:
        meta_dict["codebook_chars"] = codebook_chars
    json_path.write_text(json.dumps(meta_dict, indent=2))
    return npz_path, json_path


def load_checkpoint(path: str | Path) -> tuple[np.ndarray, np.ndarray, HymnCheckpointMetadata]:
    """Load and return (W1, W2, metadata). Frozen-flag is NOT enforced at load
    time -- that's the responsibility of any code that wants to MUTATE the weights.

    Older sidecars that predate schema_version=2 (no loss_type / context_len /
    batch_size fields) still load -- the new fields take their dataclass defaults.
    """
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    data = np.load(npz_path)
    meta_dict = json.loads(json_path.read_text())
    # Forward-compat: drop any unknown keys so future loaders don't crash here.
    known = {f.name for f in HymnCheckpointMetadata.__dataclass_fields__.values()}
    filtered = {k: v for k, v in meta_dict.items() if k in known}
    metadata = HymnCheckpointMetadata(**filtered)
    return data["W1"], data["W2"], metadata


def load_codebook(path: str | Path) -> tuple[list[str], np.ndarray] | None:
    """Load the per-checkpoint codebook if it was saved.

    Returns (chars_in_order, matrix) -- the deterministic vocab + the (V, D)
    int16 bipolar matrix the training run used. None if the checkpoint
    predates per-checkpoint codebook saving (in which case callers should
    reconstruct via `Codebook(seed=meta.seed, dim=meta.in_dim)`).
    """
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    data = np.load(npz_path)
    if "codebook_matrix" not in data:
        return None
    meta_dict = json.loads(json_path.read_text())
    chars = meta_dict.get("codebook_chars")
    if not chars:
        return None
    return chars, data["codebook_matrix"]


def freeze_checkpoint(path: str | Path) -> HymnCheckpointMetadata:
    """Mark a checkpoint as frozen by setting metadata.frozen=True and rewriting the sidecar.
    Idempotent."""
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    meta_dict = json.loads(json_path.read_text())
    meta_dict["frozen"] = True
    metadata = HymnCheckpointMetadata(**meta_dict)
    json_path.write_text(json.dumps(asdict(metadata), indent=2))
    return metadata


class FrozenCheckpointError(RuntimeError):
    """Raised when code attempts to mutate / overwrite a frozen checkpoint."""


def assert_not_frozen(metadata: HymnCheckpointMetadata) -> None:
    """Guard for code paths that intend to modify weights. Raise if frozen."""
    if metadata.frozen:
        raise FrozenCheckpointError(
            "Checkpoint is frozen. Phase 2 continual learning must not mutate HYMN weights."
        )
