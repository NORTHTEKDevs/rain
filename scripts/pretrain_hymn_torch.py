# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HYMN pre-training driver, PyTorch + DirectML edition.

Mirrors `scripts/pretrain_hymn.py` but trains on the iGPU (when DirectML
is available) with Adam + batching + optional sequence context.

Usage:

    python -m scripts.pretrain_hymn_torch \
        --corpus data/corpora/tiny_shakespeare.txt \
        --steps 50000 --batch-size 64 --context-len 8 \
        --in-dim 1024 --hidden-dim 512 \
        --out data/checkpoints/hymn_tinyshake_torch.npz

Produces the same checkpoint format as the numpy reference driver, so the
existing L1 Tier-2 benchmark consumes it without modification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rain.core.relational import Codebook
from rain.train.torch_trainer import (
    LOSS_MSE,
    LOSS_NLL,
    HymnTorch,
    auto_device,
    device_name,
    save_torch_checkpoint,
    train_torch,
)
from rain.train.warm_start_chars import warm_start_chars


def main() -> None:
    p = argparse.ArgumentParser(description="HYMN pretrain (PyTorch + DirectML)")
    p.add_argument("--corpus", required=True, help="path to text corpus")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-len", type=int, default=0,
                   help="0 = no context (matches numpy ref); K>0 bundles previous K chars")
    p.add_argument("--carry-steps", type=int, default=0,
                   help="If W>0, autoregressive sequence training: HYMN state "
                        "evolves across a W-step window with per-step loss summed. "
                        "Closer to true RNN-style training. Overrides --context-len.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Max L2 norm for gradient clipping. Recommended for "
                        "carry-steps >= 4 to prevent BPTT explosion.")
    p.add_argument("--warm-start-chars", action="store_true",
                   help="Seed the codebook with feature-based char embeddings "
                        "before training. Faster convergence to the same plateau.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW L2 regularization; helps NLL overfitting at higher step counts")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup from 0 over these steps; lets us use higher peak LR safely")
    p.add_argument("--cosine-decay", action="store_true",
                   help="After warmup, cosine-decay LR to 10%% of peak over the rest of training")
    p.add_argument("--loss", choices=[LOSS_MSE, LOSS_NLL], default=LOSS_MSE,
                   help="mse = HV-MSE (v0 reference); nll = real cross-entropy via codebook softmax")
    p.add_argument("--in-dim", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "directml"], default="auto")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--out", required=True, help="output checkpoint path (Task 2.4 format)")
    args = p.parse_args()

    if args.device == "cpu":
        import torch
        dev = torch.device("cpu")
    elif args.device == "directml":
        import torch_directml as _dml
        dev = _dml.device(0)
    else:
        dev = auto_device()
    print(f"device: {device_name(dev)}")

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    cb = Codebook(vocab_size=256, dim=args.in_dim, seed=args.seed)
    if args.warm_start_chars:
        ws_stats = warm_start_chars(cb, corpus, seed=args.seed)
        print(f"warm-start: {ws_stats}")
    model = HymnTorch(args.in_dim, args.hidden_dim, args.out_dim,
                      seed=args.seed, device=dev)

    result = train_torch(
        model, cb, corpus,
        n_steps=args.steps,
        batch_size=args.batch_size,
        context_len=args.context_len,
        carry_steps=args.carry_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cosine_decay=args.cosine_decay,
        grad_clip=args.grad_clip,
        loss_type=args.loss,
        device=dev,
        seed=args.seed,
        log_every=args.log_every,
    )

    npz_path, json_path = save_torch_checkpoint(
        model, args.out,
        n_steps=args.steps, lr=args.lr, seed=args.seed,
        initial_loss=result.initial_loss, final_loss=result.final_loss,
        losses=result.losses,
        loss_type=args.loss, context_len=args.context_len,
        carry_steps=args.carry_steps, batch_size=args.batch_size,
        codebook=cb, corpus_text=corpus,
    )
    print(f"saved {npz_path} + {json_path}")
    print(
        f"steps={result.steps}  batch={result.batch_size}  "
        f"context={result.context_len}  carry={result.carry_steps}  "
        f"loss={result.loss_type}  device={result.device}  "
        f"wall={result.wall_seconds:.1f}s  "
        f"initial_loss={result.initial_loss:.4f}  final_loss={result.final_loss:.4f}"
    )


if __name__ == "__main__":
    main()
