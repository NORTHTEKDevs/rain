# CONFIDENTIAL - PATENT PENDING
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
from dataclasses import dataclass, asdict
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


def save_checkpoint(
    path: str | Path,
    W1: np.ndarray,
    W2: np.ndarray,
    metadata: HymnCheckpointMetadata,
    losses: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Save .npz weights + .json sidecar. Returns (npz_path, json_path)."""
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    arrays = {"W1": W1.astype(np.float32), "W2": W2.astype(np.float32)}
    if losses is not None:
        arrays["losses"] = losses.astype(np.float32)
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(asdict(metadata), indent=2))
    return npz_path, json_path


def load_checkpoint(path: str | Path) -> tuple[np.ndarray, np.ndarray, HymnCheckpointMetadata]:
    """Load and return (W1, W2, metadata). Frozen-flag is NOT enforced at load
    time -- that's the responsibility of any code that wants to MUTATE the weights."""
    npz_path = Path(path).with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    data = np.load(npz_path)
    meta_dict = json.loads(json_path.read_text())
    metadata = HymnCheckpointMetadata(**meta_dict)
    return data["W1"], data["W2"], metadata


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
