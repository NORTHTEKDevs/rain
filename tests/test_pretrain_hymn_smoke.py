# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Smoke test for HYMN bootstrap pre-training: loss decreases over 10 steps on toy corpus."""

import numpy as np
import pytest
from rain.core.relational import Codebook
from scripts.pretrain_hymn import HymnSurrogate, train_steps


def test_loss_decreases_over_smoke_run():
    corpus = "the quick brown fox jumps over the lazy dog" * 50  # ~2KB
    cb = Codebook(vocab_size=64, dim=512, seed=0)
    model = HymnSurrogate(in_dim=512, hidden_dim=128, out_dim=512, seed=0)
    losses = train_steps(model, cb, corpus, n_steps=50, lr=0.01, seed=0)
    assert len(losses) == 50
    # Average loss in the last 10 steps should be lower than the first 10
    early = float(np.mean(losses[:10]))
    late = float(np.mean(losses[-10:]))
    assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"


def test_forward_shapes():
    model = HymnSurrogate(in_dim=32, hidden_dim=16, out_dim=32, seed=7)
    state = np.zeros(32, dtype=np.float32)
    input_ = np.zeros(32, dtype=np.float32)
    out, cache = model.forward(state, input_)
    assert out.shape == (32,)
    assert cache["h"].shape == (16,)
