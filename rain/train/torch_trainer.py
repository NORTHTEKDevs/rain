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
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rain.core.relational import Codebook
from rain.train.checkpoint import (
    HymnCheckpointMetadata,
    save_checkpoint,
    load_checkpoint,
)


# Loss types accepted by train_torch.
LOSS_MSE = "mse"   # MSE on bipolar hypervector targets (v0 reference, matches numpy)
LOSS_NLL = "nll"   # Cross-entropy over codebook-projected logits (real LM loss)


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


def build_char_vocab(
    corpus_text: str, codebook: Codebook, device: "torch.device"
) -> tuple[list[str], dict[str, int], torch.Tensor]:
    """Build the stable char-vocab + codebook matrix needed by NLL loss.

    Returns:
        chars_in_order: list of unique chars in the corpus, deterministic order.
        char_to_idx:    dict mapping each char to its row index in the matrix.
        codebook_matrix: (vocab, dim) float32 tensor on `device`. Each row is
                         the codebook vector for the corresponding char, ready
                         for `out @ codebook_matrix.T` to produce class logits.
    """
    chars_in_order = sorted(set(corpus_text))
    char_to_idx = {c: i for i, c in enumerate(chars_in_order)}
    matrix_np = np.stack([
        codebook.vector(c).astype(np.float32) for c in chars_in_order
    ])
    matrix = torch.from_numpy(matrix_np).to(device)
    return chars_in_order, char_to_idx, matrix


def codebook_logits(out: torch.Tensor, codebook_matrix: torch.Tensor) -> torch.Tensor:
    """Project HYMN output (batch, dim) onto the codebook vocab via dot product.

    Returns logits of shape (batch, vocab) suitable for `F.cross_entropy`.
    """
    return out @ codebook_matrix.T


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
    carry_steps: int
    loss_type: str
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
    carry_steps: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    cosine_decay: bool = False,
    grad_clip: float = 0.0,
    loss_type: str = LOSS_MSE,
    device: "torch.device | None" = None,
    seed: int = 0,
    log_every: int = 0,
) -> TorchTrainResult:
    """Train HymnTorch with AdamW on next-char prediction.

    Args:
        context_len: if 0, `input_` is zeros (matches numpy reference).
            if K>0, `input_` is the mean of the previous K bipolar codebook
            vectors at each position, projected back to float32.
        carry_steps: if 0, every batch position is an independent one-step
            next-char prediction (the v0 reference). If W>0, each batch
            sample is a window of W+1 chars; HYMN state evolves
            autoregressively across the window with per-step loss summed
            (state_{t+1} = HYMN(state_t, codebook(char_t))). This is the
            closest the v0 trainer gets to RNN-style sequence training.
            When carry_steps>0 the trainer ignores context_len (the
            evolving state already carries history).
        grad_clip: max L2-norm for gradient clipping (0 = disabled).
            Recommended for carry_steps >= 4 to prevent BPTT explosion.
        loss_type: "mse" -- MSE on the bipolar HV target (v0 reference).
            "nll" -- project HYMN output onto the codebook to produce vocab
            logits, then cross-entropy against the true next-char index.
            NLL is a much sharper gradient signal (the gap closes ~3-5x
            faster empirically) and is the real Phase-2.3 loss.
    """
    if loss_type not in (LOSS_MSE, LOSS_NLL):
        raise ValueError(f"unknown loss_type {loss_type!r}; expected 'mse' or 'nll'")
    if carry_steps < 0:
        raise ValueError(f"carry_steps must be >= 0, got {carry_steps}")
    if device is None:
        device = next(model.parameters()).device
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _lr_at(step: int) -> float:
        """Linear warmup over warmup_steps, then optional cosine decay to 10% of peak."""
        if warmup_steps and step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        if cosine_decay and n_steps > warmup_steps:
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
        return lr

    chars = list(corpus_text)
    n_corpus = len(chars)
    min_corpus = max(context_len + 2, carry_steps + 2)
    if n_corpus < min_corpus:
        raise ValueError(
            f"corpus too short ({n_corpus} chars) for "
            f"context_len={context_len} carry_steps={carry_steps}"
        )

    # Pre-build NLL machinery once if applicable (much faster than per-step).
    if loss_type == LOSS_NLL:
        _, char_to_idx, codebook_matrix = build_char_vocab(corpus_text, codebook, device)
    else:
        char_to_idx, codebook_matrix = None, None

    rng = np.random.default_rng(seed)
    losses: list[float] = []

    t0 = time.perf_counter()
    for step in range(n_steps):
        if warmup_steps or cosine_decay:
            for g in optim.param_groups:
                g["lr"] = _lr_at(step)

        if carry_steps > 0:
            # Sequence-carry training: each batch sample is a (carry_steps+1)-char
            # window. We RNN-evolve HYMN state with teacher forcing:
            #   state = ZERO (neutral hidden state)
            #   for t in 0..carry_steps:
            #       state = HYMN(state, codebook(char_t))   # ingest current char
            #       predict char_{t+1} from state           # then decode
            # Loss is averaged over the carry_steps predictions (one per window
            # position except the last input-only step).
            high = max(carry_steps + 1, n_corpus - (carry_steps + 2))
            starts = rng.integers(0, high, size=batch_size)
            window_chars = [
                [chars[s + i] for i in range(carry_steps + 2)]
                for s in starts
            ]
            state = torch.zeros((batch_size, model.in_dim), device=device)
            total_loss = torch.zeros((), device=device)
            for t in range(carry_steps + 1):
                cur_chars = [w[t] for w in window_chars]
                input_vec = _codebook_lookup_batch(codebook, cur_chars, device)
                state = model(state, input_vec)
                # Predict char_{t+1} from the updated state.
                target_chars = [w[t + 1] for w in window_chars]
                if loss_type == LOSS_NLL:
                    target_idx = torch.tensor(
                        [char_to_idx[c] for c in target_chars],
                        dtype=torch.long, device=device,
                    )
                    logits = codebook_logits(state, codebook_matrix)
                    total_loss = total_loss + F.cross_entropy(logits, target_idx)
                else:
                    target_vec = _codebook_lookup_batch(codebook, target_chars, device)
                    total_loss = total_loss + ((state - target_vec) ** 2).mean()
            loss = total_loss / float(carry_steps + 1)
        else:
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
            out = model(state, input_)
            if loss_type == LOSS_NLL:
                target_idx = torch.tensor(
                    [char_to_idx[c] for c in next_chars], dtype=torch.long, device=device,
                )
                logits = codebook_logits(out, codebook_matrix)
                loss = F.cross_entropy(logits, target_idx)
            else:
                target = _codebook_lookup_batch(codebook, next_chars, device)
                loss = ((out - target) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        losses.append(float(loss.detach().cpu()))

        if log_every and (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-min(log_every, len(losses)):]))
            unit = "nats/char" if loss_type == LOSS_NLL else "hv_mse"
            print(
                f"step {step + 1}/{n_steps}  loss(avg last {log_every}, {unit}) = {recent:.4f}",
                flush=True,  # vital when stdout is a pipe (block-buffered otherwise)
            )

    wall = time.perf_counter() - t0
    return TorchTrainResult(
        steps=n_steps,
        batch_size=batch_size,
        context_len=context_len,
        carry_steps=carry_steps,
        loss_type=loss_type,
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
    loss_type: str = LOSS_MSE,
    context_len: int = 0,
    carry_steps: int = 0,
    batch_size: int = 1,
    codebook: "Codebook | None" = None,
    corpus_text: str | None = None,
) -> tuple[Path, Path]:
    """Detach weights + write the numpy checkpoint format.

    Carries loss_type + context_len + carry_steps + batch_size into the
    sidecar so consumers (L1 benchmark, future scripts) can auto-pick the
    right eval metric and eval mode.
    """
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
        schema_version=2,
        loss_type=loss_type,
        context_len=context_len,
        carry_steps=carry_steps,
        batch_size=batch_size,
    )
    codebook_chars = None
    codebook_matrix = None
    if codebook is not None and corpus_text is not None:
        # Persist the exact codebook entries for every char in the corpus
        # so inference reproduces what training saw -- essential for
        # warm-started training where the codebook isn't a pure function
        # of (seed, dim).
        codebook_chars = sorted(set(corpus_text))
        codebook_matrix = np.stack(
            [codebook.vector(c).astype(np.int16) for c in codebook_chars]
        )
    return save_checkpoint(
        path, W1, W2, meta,
        losses=np.asarray(losses, dtype=np.float32) if losses else None,
        codebook_chars=codebook_chars,
        codebook_matrix=codebook_matrix,
    )
