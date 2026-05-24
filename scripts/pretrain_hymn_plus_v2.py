# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Training driver for HYMN-Plus v2 (BPE + KB-Attention).

Trains a BPE tokenizer on the corpus (or loads a saved one), encodes
the corpus into BPE tokens, then trains HymnPlusV2 with the standard
sequence-carry recipe + held-out validation.

Usage:
    python -m scripts.pretrain_hymn_plus_v2 \
        --corpus data/corpora/tiny_shakespeare.txt \
        --bpe-vocab 2048 \
        --steps 5000 --batch-size 16 --seq-len 96 \
        --dim 192 --n-layers 4 --mlp-mult 4 \
        --kb-size 1024 --kb-top-k 8 \
        --lr 3e-4 --warmup-steps 400 --cosine-decay --weight-decay 0.05 \
        --grad-clip 1.0 \
        --val-split 0.05 --val-every 500 \
        --device cpu --log-every 500 \
        --out data/checkpoints/hymn_plus_v2_bpe.npz
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config, count_params
from rain.tokenize.bpe import BPETokenizer


def _lr_schedule(step: int, total: int, peak_lr: float, warmup: int, cosine: bool) -> float:
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    if not cosine:
        return peak_lr
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (0.1 + 0.9 * cos)


@torch.no_grad()
def _eval_val(model, val_ids, *, batch_size, seq_len, device, max_windows=64):
    was_training = model.training
    model.eval()
    n_avail = (len(val_ids) - 1) // seq_len
    n_windows = max(batch_size, (min(max_windows, n_avail) // batch_size) * batch_size)
    rng = np.random.default_rng(0)
    total_loss = 0.0
    total_chars = 0
    for _ in range(0, n_windows, batch_size):
        starts = rng.integers(0, len(val_ids) - seq_len - 1, size=batch_size)
        windows = np.stack([val_ids[s : s + seq_len + 1] for s in starts])
        x = torch.as_tensor(windows[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(windows[:, 1:], device=device, dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_chars += y.numel()
    model.train(was_training)
    return total_loss / max(1, total_chars)


def train(
    model,
    train_ids,
    val_ids,
    *,
    n_steps,
    batch_size,
    seq_len,
    lr,
    weight_decay,
    warmup_steps,
    cosine_decay,
    grad_clip,
    device,
    seed,
    log_every,
    val_every,
    early_stop_patience,
):
    model.to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    N = len(train_ids)
    losses = []
    val_hist = []
    initial_loss = None
    best_val = float("inf")
    n_no_imp = 0
    stopped_early = None
    start = time.time()

    val_interval = val_every if val_every > 0 else max(1, n_steps // 10)

    for step in range(n_steps):
        cur_lr = _lr_schedule(step, n_steps, lr, warmup_steps, cosine_decay)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        starts = rng.integers(0, N - seq_len - 1, size=batch_size)
        batch = np.stack([train_ids[s : s + seq_len + 1] for s in starts])
        x = torch.as_tensor(batch[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(batch[:, 1:], device=device, dtype=torch.long)

        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if initial_loss is None:
            initial_loss = loss_val

        if (step + 1) % log_every == 0:
            window = losses[-log_every:]
            avg = sum(window) / len(window)
            msg = (
                f"step {step + 1}/{n_steps}  "
                f"train(avg last {log_every}, nats/token) = {avg:.4f}  "
                f"lr={cur_lr:.2e}"
            )
            if val_ids is not None and (step + 1) % val_interval == 0:
                vnll = _eval_val(
                    model, val_ids, batch_size=batch_size, seq_len=seq_len, device=device
                )
                val_hist.append((step + 1, vnll))
                gap = vnll - avg
                msg += f"  val = {vnll:.4f}  gap = {gap:+.4f}"
                if vnll < best_val - 0.001:
                    best_val = vnll
                    n_no_imp = 0
                else:
                    n_no_imp += 1
                    if early_stop_patience > 0 and n_no_imp >= early_stop_patience:
                        msg += f"  EARLY STOP after {n_no_imp} no-improvement checks"
                        print(msg)
                        stopped_early = step + 1
                        break
            print(msg)
        if stopped_early is not None:
            break

    wall = time.time() - start
    final = float(sum(losses[-min(1000, len(losses)) :]) / min(1000, len(losses)))
    return {
        "steps": n_steps,
        "stopped_early_at": stopped_early,
        "wall_seconds": wall,
        "initial_loss": initial_loss,
        "final_loss": final,
        "final_val_loss": val_hist[-1][1] if val_hist else None,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "val_history": val_hist,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HYMN-Plus v2 pretrain (BPE + KB-attention)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--bpe-vocab", type=int, default=2048)
    p.add_argument(
        "--bpe-model",
        default=None,
        help="path to pre-trained sentencepiece .model; if omitted, train inline",
    )
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-tie-weights", action="store_true")
    p.add_argument("--kb-size", type=int, default=1024)
    p.add_argument("--kb-top-k", type=int, default=8)
    p.add_argument(
        "--kb-attn-in-layers",
        type=int,
        nargs="*",
        default=None,
        help="indices of blocks that get KB-attention (default: all)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu"], default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = torch.device("cpu")
    print(f"device: {dev}")

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    print(f"corpus: {len(corpus):,} chars")

    # BPE: train or load
    tok = BPETokenizer(vocab_size=args.bpe_vocab)
    if args.bpe_model:
        tok.load(args.bpe_model)
        print(f"loaded BPE model from {args.bpe_model}")
    else:
        print(f"training BPE (vocab={args.bpe_vocab}) ...")
        tok.train([corpus])
    ids = np.array(tok.encode(corpus), dtype=np.int64)
    print(
        f"corpus encoded: {len(ids):,} tokens (compression {len(corpus) / len(ids):.2f} chars/tok)"
    )
    actual_vocab = tok._sp.get_piece_size()

    # Val split (contiguous tail)
    if args.val_split > 0.0:
        n_val = max(args.seq_len * 8, int(len(ids) * args.val_split))
        n_val = min(n_val, len(ids) // 4)
        train_ids = ids[:-n_val]
        val_ids = ids[-n_val:]
        print(
            f"split: train={len(train_ids):,} tokens, val={len(val_ids):,} tokens "
            f"({args.val_split * 100:.1f}%)"
        )
    else:
        train_ids = ids
        val_ids = None

    cfg = HymnPlusV2Config(
        vocab_size=actual_vocab,
        dim=args.dim,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
        kb_size=args.kb_size,
        kb_top_k=args.kb_top_k,
        kb_attn_in_layers=tuple(args.kb_attn_in_layers) if args.kb_attn_in_layers else None,
        seed=args.seed,
    )
    model = HymnPlusV2(cfg)
    print(f"HYMN-Plus v2 trainable params: {count_params(model):,}")
    print(
        f"KB buffer: {cfg.kb_size} facts x dim={cfg.dim} = "
        f"{cfg.kb_size * cfg.dim:,} floats (non-trainable)"
    )

    result = train(
        model,
        train_ids,
        val_ids,
        n_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cosine_decay=args.cosine_decay,
        grad_clip=args.grad_clip,
        device=dev,
        seed=args.seed,
        log_every=args.log_every,
        val_every=args.val_every,
        early_stop_patience=args.early_stop_patience,
    )

    # Save checkpoint
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, **sd)

    # Also persist the BPE model alongside the checkpoint
    bpe_path = out.with_suffix(".bpe.model")
    tok.save(bpe_path)

    meta = {
        "arch": "hymn_plus_v2",
        "dim": args.dim,
        "n_layers": args.n_layers,
        "mlp_mult": args.mlp_mult,
        "vocab_size": actual_vocab,
        "bpe_vocab": args.bpe_vocab,
        "bpe_model_path": str(bpe_path),
        "kb_size": args.kb_size,
        "kb_top_k": args.kb_top_k,
        "kb_attn_in_layers": cfg.kb_attn_in_layers,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "tie_weights": not args.no_tie_weights,
        "trainable_params": count_params(model),
        "wall_seconds": result["wall_seconds"],
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "final_val_loss": result.get("final_val_loss"),
        "best_val_loss": result.get("best_val_loss"),
        "stopped_early_at": result.get("stopped_early_at"),
        "val_history": result.get("val_history", []),
        "val_split": args.val_split,
        "early_stop_patience": args.early_stop_patience,
        "device": str(dev),
        "seed": args.seed,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {out} + {out.with_suffix('.json')} + {bpe_path}")
    val_str = (
        f"  val={result['final_val_loss']:.4f}" if result.get("final_val_loss") is not None else ""
    )
    print(
        f"steps={result['steps']}  wall={result['wall_seconds']:.1f}s  "
        f"initial_loss={result['initial_loss']:.4f}  "
        f"final_loss={result['final_loss']:.4f}{val_str}"
    )


if __name__ == "__main__":
    main()
