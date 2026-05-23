# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
import pytest
import torch

from rain.core.relational import Codebook
from rain.train.torch_trainer import (
    HymnTorch,
    train_torch,
    save_torch_checkpoint,
)
from rain.train.checkpoint import load_checkpoint
from scripts.pretrain_hymn import HymnSurrogate


CPU = torch.device("cpu")


def test_init_matches_numpy_reference():
    """Same seed -> HymnTorch and HymnSurrogate produce identical W1, W2."""
    in_dim, hidden, out_dim = 32, 16, 32
    np_model = HymnSurrogate(in_dim, hidden, out_dim, seed=7)
    pt_model = HymnTorch(in_dim, hidden, out_dim, seed=7, device=CPU)
    assert np.allclose(np_model.W1, pt_model.W1.detach().numpy())
    assert np.allclose(np_model.W2, pt_model.W2.detach().numpy())


def test_forward_matches_numpy_reference():
    """At identical weights and input, both implementations produce identical out."""
    in_dim, hidden, out_dim = 32, 16, 32
    np_model = HymnSurrogate(in_dim, hidden, out_dim, seed=0)
    pt_model = HymnTorch(in_dim, hidden, out_dim, seed=0, device=CPU)
    state = np.random.default_rng(1).standard_normal(in_dim).astype(np.float32)
    input_ = np.zeros_like(state)
    np_out, _ = np_model.forward(state, input_)
    pt_out = pt_model(
        torch.from_numpy(state).unsqueeze(0),
        torch.from_numpy(input_).unsqueeze(0),
    ).detach().numpy()[0]
    assert np.allclose(np_out, pt_out, atol=1e-6), (
        f"forward divergence: max diff = {np.max(np.abs(np_out - pt_out))}"
    )


def test_train_loop_reduces_loss_on_cpu():
    """Smoke: 200 steps on a toy corpus on CPU should reduce MSE-on-HV."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r = train_torch(
        model, cb, corpus,
        n_steps=200, batch_size=16, context_len=0,
        lr=5e-3, device=CPU, seed=0,
    )
    assert len(r.losses) == 200
    early = float(np.mean(r.losses[:20]))
    late = float(np.mean(r.losses[-20:]))
    assert late < early, f"loss did not decrease on CPU: early={early}, late={late}"


def test_context_len_changes_training_dynamics():
    """Sequence context should produce a different (and usually better) loss
    trajectory than context-len=0 on the same seed."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    cb_a = Codebook(vocab_size=64, dim=128, seed=0)
    cb_b = Codebook(vocab_size=64, dim=128, seed=0)
    m_a = HymnTorch(128, 64, 128, seed=0, device=CPU)
    m_b = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r_a = train_torch(m_a, cb_a, corpus, n_steps=100, batch_size=8,
                      context_len=0, lr=5e-3, device=CPU, seed=0)
    r_b = train_torch(m_b, cb_b, corpus, n_steps=100, batch_size=8,
                      context_len=4, lr=5e-3, device=CPU, seed=0)
    # Different trajectories on otherwise identical setup
    assert r_a.losses != r_b.losses


def test_checkpoint_roundtrip(tmp_path):
    """Train briefly, save via save_torch_checkpoint, load back with HymnTorch.from_checkpoint."""
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    corpus = "the quick brown fox " * 80
    r = train_torch(model, cb, corpus, n_steps=50, batch_size=8,
                    context_len=0, lr=5e-3, device=CPU, seed=0)
    ckpt_path = tmp_path / "test_ckpt.npz"
    save_torch_checkpoint(model, ckpt_path, n_steps=50, lr=5e-3, seed=0,
                          initial_loss=r.initial_loss, final_loss=r.final_loss)
    # Load back as plain numpy (matches L1 benchmark loader)
    W1, W2, meta = load_checkpoint(ckpt_path)
    assert W1.shape == (128, 64)
    assert W2.shape == (64, 128)
    assert meta.in_dim == 128 and meta.hidden_dim == 64 and meta.out_dim == 128
    # And as HymnTorch
    loaded, meta2 = HymnTorch.from_checkpoint(ckpt_path, device=CPU)
    assert np.allclose(loaded.W1.detach().numpy(), model.W1.detach().numpy())
    assert np.allclose(loaded.W2.detach().numpy(), model.W2.detach().numpy())
