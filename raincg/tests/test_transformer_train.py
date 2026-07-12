import tempfile
import os
import torch
from raincg.bench.transformer_baseline import (
    train_baseline, load_checkpoint, greedy_decode,
)


def test_overfits_tiny_set():
    pairs = [("copy A B", ["A", "B"]), ("reverse A B C", ["C", "B", "A"]),
             ("echo X", ["X", "X"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        res = train_baseline(pairs, d_model=32, nhead=2, num_layers=1, ff=64,
                             epochs=80, lr=1e-3, batch_size=3, ckpt_path=ckpt,
                             max_minutes=2.0)
        assert res.params > 0
        model, vocab = load_checkpoint(ckpt)
        got = greedy_decode(model, vocab, "copy A B", max_len=10)
        assert got == ["A", "B"]


def test_checkpoint_saved_every_epoch_has_resume_fields():
    pairs = [("copy A B", ["A", "B"]), ("reverse A B C", ["C", "B", "A"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                       epochs=2, lr=1e-3, batch_size=2, ckpt_path=ckpt,
                       max_minutes=5.0)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck["epoch"] == 2
        assert "optimizer" in ck
        assert ck["elapsed_train_s"] > 0.0


def test_checkpoint_resume_round_trip_continues_epoch_counter():
    pairs = [("copy A B", ["A", "B"]), ("reverse A B C", ["C", "B", "A"]),
             ("echo X", ["X", "X"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        # simulate a killed run: only 1 of 4 planned epochs completes
        r1 = train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                            epochs=1, lr=1e-3, batch_size=3, ckpt_path=ckpt,
                            max_minutes=5.0)
        assert r1.epochs_run == 1
        elapsed_after_1 = r1.train_seconds

        # resume: should continue at epoch 2, not restart at epoch 1
        r2 = train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                            epochs=4, lr=1e-3, batch_size=3, ckpt_path=ckpt,
                            max_minutes=5.0, resume=True)
        assert r2.epochs_run == 4
        # cumulative train_s must include prior elapsed time, not reset to 0
        assert r2.train_seconds >= elapsed_after_1

        model, vocab = load_checkpoint(ckpt)
        # param shape (vocab x d_model) unchanged across resume
        assert model.emb.weight.shape[1] == 16
        got = greedy_decode(model, vocab, "copy A B", max_len=10)
        assert isinstance(got, list)


def test_checkpoint_resume_noop_when_already_complete():
    pairs = [("copy A B", ["A", "B"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                       epochs=3, lr=1e-3, batch_size=1, ckpt_path=ckpt,
                       max_minutes=5.0)
        r2 = train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                            epochs=3, lr=1e-3, batch_size=1, ckpt_path=ckpt,
                            max_minutes=5.0, resume=True)
        # already at target epoch count; resume should not re-run epochs
        assert r2.epochs_run == 3


def test_resume_without_existing_checkpoint_starts_fresh():
    pairs = [("copy A B", ["A", "B"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        res = train_baseline(pairs, d_model=16, nhead=2, num_layers=1, ff=32,
                             epochs=2, lr=1e-3, batch_size=1, ckpt_path=ckpt,
                             max_minutes=5.0, resume=True)
        assert res.epochs_run == 2
