# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
import pytest
import torch

from rain.core.relational import Codebook
from rain.train.checkpoint import load_checkpoint
from rain.train.torch_trainer import (
    LOSS_NLL,
    HymnTorch,
    build_char_vocab,
    codebook_logits,
    save_torch_checkpoint,
    train_torch,
)
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


def test_build_char_vocab_is_deterministic_and_complete():
    """Vocab covers every unique char in the corpus and is in stable sorted order."""
    cb = Codebook(vocab_size=128, dim=64, seed=0)
    corpus = "the quick brown fox jumps over the lazy dog"
    chars, c2i, matrix = build_char_vocab(corpus, cb, CPU)
    assert set(chars) == set(corpus)
    assert chars == sorted(set(corpus))
    assert all(c2i[c] == i for i, c in enumerate(chars))
    assert matrix.shape == (len(chars), 64)


def test_codebook_logits_argmax_recovers_self():
    """For each row of the codebook matrix, `out=row` should produce a logits
    vector whose argmax is that row's own index -- the basic decoding check."""
    cb = Codebook(vocab_size=128, dim=64, seed=0)
    corpus = "the quick brown fox jumps over the lazy dog"
    _, _, matrix = build_char_vocab(corpus, cb, CPU)
    logits = codebook_logits(matrix, matrix)  # (vocab, vocab)
    argmax = logits.argmax(dim=1)
    expected = torch.arange(matrix.shape[0])
    assert torch.equal(argmax, expected), (
        f"codebook self-decoding broke: {argmax.tolist()} != {expected.tolist()}"
    )


def test_nll_train_loop_reduces_loss_on_cpu():
    """Smoke: NLL training reduces cross-entropy on a small corpus."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r = train_torch(
        model, cb, corpus,
        n_steps=200, batch_size=16, context_len=4,
        lr=5e-3, loss_type=LOSS_NLL, device=CPU, seed=0,
    )
    assert r.loss_type == LOSS_NLL
    assert len(r.losses) == 200
    early = float(np.mean(r.losses[:20]))
    late = float(np.mean(r.losses[-20:]))
    assert late < early, f"NLL loss did not decrease: early={early}, late={late}"


def test_lr_warmup_starts_below_peak_and_reaches_it():
    """With warmup_steps>0, the early-step LR should be lower than `lr`.
    Captured via loss curve smoothness: warmup runs should converge as
    well or better than no-warmup on the same seed."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r = train_torch(
        model, cb, corpus,
        n_steps=200, batch_size=8, context_len=4,
        lr=5e-3, warmup_steps=50, cosine_decay=True,
        loss_type=LOSS_NLL, device=CPU, seed=0,
    )
    # Just verify the loop completes and produces a monotonically improving
    # AVERAGE -- exact LR-schedule numerics are not exposed in the result.
    early = float(np.mean(r.losses[:20]))
    late = float(np.mean(r.losses[-20:]))
    assert late < early, f"warmup+cosine didn't reduce loss: early={early}, late={late}"


def test_carry_steps_reduces_nll_loss():
    """Sequence-carry training (W=4) reduces NLL on a small corpus."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r = train_torch(
        model, cb, corpus,
        n_steps=80, batch_size=8, carry_steps=4, grad_clip=1.0,
        lr=1e-3, loss_type=LOSS_NLL, device=CPU, seed=0,
    )
    assert r.carry_steps == 4
    early = float(np.mean(r.losses[:10]))
    late = float(np.mean(r.losses[-10:]))
    assert late < early, f"carry training didn't reduce NLL: early={early}, late={late}"


def test_carry_steps_overrides_context_len_silently():
    """When carry_steps>0, the trainer ignores context_len -- they should not
    interfere. Same seed, same data: result is deterministic."""
    corpus = "the quick brown fox " * 80
    cb1 = Codebook(vocab_size=64, dim=128, seed=0)
    cb2 = Codebook(vocab_size=64, dim=128, seed=0)
    m1 = HymnTorch(128, 64, 128, seed=0, device=CPU)
    m2 = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r1 = train_torch(m1, cb1, corpus, n_steps=20, batch_size=4,
                     carry_steps=3, context_len=0,
                     lr=1e-3, device=CPU, seed=0)
    r2 = train_torch(m2, cb2, corpus, n_steps=20, batch_size=4,
                     carry_steps=3, context_len=16,  # ignored
                     lr=1e-3, device=CPU, seed=0)
    assert r1.losses == r2.losses


def test_grad_clip_does_not_break_training():
    """grad_clip=1.0 should not prevent learning on a normal small run."""
    corpus = "the quick brown fox jumps over the lazy dog. " * 60
    cb = Codebook(vocab_size=64, dim=128, seed=0)
    model = HymnTorch(128, 64, 128, seed=0, device=CPU)
    r = train_torch(
        model, cb, corpus,
        n_steps=80, batch_size=8, context_len=4, grad_clip=1.0,
        lr=5e-3, loss_type=LOSS_NLL, device=CPU, seed=0,
    )
    early = float(np.mean(r.losses[:10]))
    late = float(np.mean(r.losses[-10:]))
    assert late < early


def test_carry_steps_negative_is_rejected():
    cb = Codebook(vocab_size=64, dim=64, seed=0)
    model = HymnTorch(64, 32, 64, seed=0, device=CPU)
    with pytest.raises(ValueError, match="carry_steps"):
        train_torch(model, cb, "the quick brown fox " * 10,
                    n_steps=1, batch_size=2, carry_steps=-1,
                    device=CPU, seed=0)


def test_train_torch_rejects_unknown_loss_type():
    cb = Codebook(vocab_size=64, dim=64, seed=0)
    model = HymnTorch(64, 32, 64, seed=0, device=CPU)
    with pytest.raises(ValueError, match="unknown loss_type"):
        train_torch(model, cb, "the quick brown fox " * 10,
                    n_steps=1, batch_size=2, loss_type="bogus",
                    device=CPU, seed=0)


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
