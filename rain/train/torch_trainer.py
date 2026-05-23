# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""PyTorch HYMN trainer with optional DirectML acceleration.

Mirrors the forward semantics of `scripts.pretrain_hymn.HymnSurrogate` so a
checkpoint trained here loads cleanly via `rain.train.checkpoint` and is
consumed by `evals.tier2_llm_parity.tiny_shakespeare`.

Why this module exists:
- `HymnSurrogate` (numpy) is the deterministic reference implementation but
  runs single-example SGD on CPU only. On the Corsair AI Workstation 300
  this leaves the iGPU idle and tops out at ~1000 steps/sec.
- `HymnTorch` (this module) keeps the architecture bit-for-bit identical
  to the numpy version at the same random seed, but uses PyTorch autograd
  + Adam + DirectML so the same hardware delivers ~5-10x throughput on the
  Radeon 8060S.
- Optional sequence-context bundling lets the model see prior-K chars at
  each step (the single biggest L1 quality lever).

Auto-detects DirectML if `torch_directml` is importable; otherwise CPU.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rain.core.relational import Codebook
from rain.train.checkpoint import (
    HymnCheckpointMetadata,
    save_checkpoint,
    load_checkpoint,
)


def auto_device() -> "torch.device":
    """Return the best available training device on this machine.

    Order: directml (AMD iGPU/dGPU on Windows) -> cuda -> mps -> cpu.
    The directml path is what the Corsair AI Workstation 300 uses.
    """
    try:
        import torch_directml as _dml
        if _dml.device_count() > 0:
            return _dml.device(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: "torch.device | Any") -> str:
    """Human-readable device label for logging."""
    try:
        import torch_directml as _dml
        # DirectML returns a string-formattable device; probe via dml api
        if "privateuseone" in str(dev).lower() or "dml" in str(dev).lower():
            return f"DirectML({_dml.device_name(0)})"
    except Exception:
        pass
    return str(dev)


class HymnTorch(nn.Module):
    """PyTorch port of `HymnSurrogate`.

    Forward: `out = tanh(tanh(state + input_) @ W1 -> tanh @ W2)`.
    Initialization matches the numpy version exactly at the same seed so
    a randomly-init checkpoint from either side reproduces identically.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        seed: int = 42,
        device: "torch.device | None" = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        W1 = (rng.standard_normal((in_dim, hidden_dim)) / np.sqrt(in_dim)).astype(np.float32)
        W2 = (rng.standard_normal((hidden_dim, out_dim)) / np.sqrt(hidden_dim)).astype(np.float32)
        self.W1 = nn.Parameter(torch.from_numpy(W1))
        self.W2 = nn.Parameter(torch.from_numpy(W2))
        if device is not None:
            self.to(device)

    def forward(self, state: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        """Tensor shapes: (batch, in_dim) for state + input_; returns (batch, out_dim)."""
        combined = torch.tanh(state + input_)
        h = torch.tanh(combined @ self.W1)
        out = torch.tanh(h @ self.W2)
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: "str | Path",
        device: "torch.device | None" = None,
    ) -> tuple["HymnTorch", HymnCheckpointMetadata]:
        """Load weights + metadata from a numpy checkpoint into a HymnTorch."""
        W1, W2, meta = load_checkpoint(path)
        model = cls(meta.in_dim, meta.hidden_dim, meta.out_dim, seed=meta.seed, device=device)
        with torch.no_grad():
            target_device = model.W1.device
            model.W1.copy_(torch.from_numpy(W1).to(target_device))
            model.W2.copy_(torch.from_numpy(W2).to(target_device))
        return model, meta

    def to_numpy_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Detach + move-to-CPU + numpyify for checkpoint save."""
        return (
            self.W1.detach().cpu().numpy().astype(np.float32),
            self.W2.detach().cpu().numpy().astype(np.float32),
        )


def _codebook_lookup_batch(
    codebook: Codebook, chars: list[str], device: "torch.device"
) -> torch.Tensor:
    """Materialize a (batch, D) float32 tensor from a list of chars."""
    arr = np.stack([codebook.vector(c).astype(np.float32) for c in chars])
    return torch.from_numpy(arr).to(device)


def _sample_positions(
    n_corpus: int, batch_size: int, context_len: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample (batch_size,) start positions ensuring at least context_len chars
    of history are available."""
    low = context_len
    high = max(low + 1, n_corpus - 1)
    return rng.integers(low, high, size=batch_size)


@dataclass
class TorchTrainResult:
    steps: int
    batch_size: int
    context_len: int
    initial_loss: float
    final_loss: float
    losses: list[float]
    wall_seconds: float
    device: str
    notes: list[str] = field(default_factory=list)


def train_torch(
    model: HymnTorch,
    codebook: Codebook,
    corpus_text: str,
    *,
    n_steps: int,
    batch_size: int = 32,
    context_len: int = 0,
    lr: float = 1e-3,
    device: "torch.device | None" = None,
    seed: int = 0,
    log_every: int = 0,
) -> TorchTrainResult:
    """Train HymnTorch with Adam on next-char MSE-on-HV.

    Args:
        context_len: if 0, `input_` is zeros (matches numpy reference).
            if K>0, `input_` is the mean of the previous K bipolar codebook
            vectors at each position, projected back to float32. Equivalent
            to a simple bundle of the recent history.
    """
    if device is None:
        device = next(model.parameters()).device
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    chars = list(corpus_text)
    n_corpus = len(chars)
    if n_corpus < context_len + 2:
        raise ValueError(
            f"corpus too short ({n_corpus} chars) for context_len={context_len}"
        )

    rng = np.random.default_rng(seed)
    losses: list[float] = []

    t0 = time.perf_counter()
    for step in range(n_steps):
        positions = _sample_positions(n_corpus, batch_size, context_len, rng)
        cur_chars = [chars[p] for p in positions]
        next_chars = [chars[p + 1] for p in positions]
        state = _codebook_lookup_batch(codebook, cur_chars, device)
        if context_len > 0:
            history_stack = np.zeros((batch_size, context_len, model.in_dim), dtype=np.float32)
            for bi, p in enumerate(positions):
                for ki in range(context_len):
                    prev_pos = p - (ki + 1)
                    if prev_pos >= 0:
                        history_stack[bi, ki] = codebook.vector(chars[prev_pos]).astype(np.float32)
            history_mean = history_stack.mean(axis=1)
            input_ = torch.from_numpy(history_mean).to(device)
        else:
            input_ = torch.zeros_like(state)
        target = _codebook_lookup_batch(codebook, next_chars, device)

        out = model(state, input_)
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))

        if log_every and (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-min(log_every, len(losses)):]))
            print(f"step {step + 1}/{n_steps}  loss(avg last {log_every}) = {recent:.4f}")

    wall = time.perf_counter() - t0
    return TorchTrainResult(
        steps=n_steps,
        batch_size=batch_size,
        context_len=context_len,
        initial_loss=float(losses[0]) if losses else 0.0,
        final_loss=float(losses[-1]) if losses else 0.0,
        losses=losses,
        wall_seconds=wall,
        device=device_name(device),
    )


def save_torch_checkpoint(
    model: HymnTorch,
    path: "str | Path",
    *,
    n_steps: int,
    lr: float,
    seed: int,
    initial_loss: float | None,
    final_loss: float | None,
    losses: list[float] | None = None,
) -> tuple[Path, Path]:
    """Detach weights + write the numpy checkpoint format."""
    W1, W2 = model.to_numpy_weights()
    meta = HymnCheckpointMetadata(
        in_dim=model.in_dim,
        hidden_dim=model.hidden_dim,
        out_dim=model.out_dim,
        steps=n_steps,
        lr=lr,
        seed=seed,
        final_loss=final_loss,
        initial_loss=initial_loss,
    )
    return save_checkpoint(
        path, W1, W2, meta,
        losses=np.asarray(losses, dtype=np.float32) if losses else None,
    )
