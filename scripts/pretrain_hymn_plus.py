# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Training driver for HYMN-Plus.

Same recipe as pretrain_hymn_torch (NLL loss, AdamW + warmup + cosine,
sequence-carry training) so the A/B comparison is clean.

Usage:
    python -m scripts.pretrain_hymn_plus \
        --corpus data/corpora/tiny_shakespeare.txt \
        --steps 15000 --batch-size 16 --seq-len 64 \
        --dim 256 --n-layers 4 --mlp-mult 4 \
        --lr 3e-4 --warmup-steps 500 --cosine-decay --weight-decay 0.05 \
        --grad-clip 1.0 --device directml \
        --out data/checkpoints/hymn_plus_v1_15k.npz
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

from rain.core.hymn_plus import HymnPlus, HymnPlusConfig, count_params
from rain.core.relational import Codebook
from rain.train.warm_start_chars import warm_start_chars


def _auto_device() -> torch.device:
    try:
        import torch_directml as _dml

        return _dml.device(0)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _device_name(dev: torch.device) -> str:
    name = str(dev)
    if "privateuse" in name or "directml" in name.lower():
        try:
            import torch_directml as _dml

            return f"DirectML({_dml.device_name(0)})"
        except Exception:
            return "DirectML"
    return name


def _build_char_vocab(text: str) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(set(text)))}


def _bipolar_codebook(rain_cb: Codebook, char_to_id: dict[str, int], dim: int) -> np.ndarray:
    rows = []
    for ch, _ in sorted(char_to_id.items(), key=lambda kv: kv[1]):
        v = (
            rain_cb.vector(ch)[:dim]
            if rain_cb.dim >= dim
            else np.tile(rain_cb.vector(ch), (dim // rain_cb.dim + 1))[:dim]
        )
        rows.append(v)
    return np.stack(rows, axis=0).astype(np.int16)


def _lr_schedule(step: int, total: int, peak_lr: float, warmup: int, cosine: bool) -> float:
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    if not cosine:
        return peak_lr
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (0.1 + 0.9 * cos)


def train(
    model: HymnPlus,
    corpus: str,
    char_to_id: dict[str, int],
    *,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    cosine_decay: bool,
    grad_clip: float,
    device: torch.device,
    seed: int,
    log_every: int,
) -> dict:
    model.to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    ids = np.fromiter((char_to_id[c] for c in corpus if c in char_to_id), dtype=np.int64)
    N = len(ids)
    losses: list[float] = []
    initial_loss = None
    start = time.time()

    for step in range(n_steps):
        cur_lr = _lr_schedule(step, n_steps, lr, warmup_steps, cosine_decay)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        starts = rng.integers(0, N - seq_len - 1, size=batch_size)
        batch = np.stack([ids[s : s + seq_len + 1] for s in starts], axis=0)
        x = torch.as_tensor(batch[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(batch[:, 1:], device=device, dtype=torch.long)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="mean",
        )
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
            print(
                f"step {step + 1}/{n_steps}  "
                f"loss(avg last {log_every}, nats/char) = {avg:.4f}  "
                f"lr={cur_lr:.2e}"
            )

    wall = time.time() - start
    final = float(sum(losses[-min(1000, len(losses)) :]) / min(1000, len(losses)))
    return {
        "steps": n_steps,
        "wall_seconds": wall,
        "initial_loss": initial_loss,
        "final_loss": final,
        "losses": losses,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HYMN-Plus pretrain (PyTorch + DirectML)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warm-start-chars", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "directml"], default="auto")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = (
        _auto_device()
        if args.device == "auto"
        else (torch.device("cpu") if args.device == "cpu" else _auto_device())
    )
    print(f"device: {_device_name(dev)}")

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    char_to_id = _build_char_vocab(corpus)
    print(f"corpus: {len(corpus):,} chars, {len(char_to_id)} unique")

    init_cb = None
    if args.warm_start_chars:
        rain_cb = Codebook(vocab_size=256, dim=args.dim, seed=args.seed)
        ws = warm_start_chars(rain_cb, corpus, seed=args.seed)
        print(f"warm-start[features]: {ws}")
        init_cb = _bipolar_codebook(rain_cb, char_to_id, args.dim)

    cfg = HymnPlusConfig(
        vocab_size=len(char_to_id),
        dim=args.dim,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        init_codebook=init_cb,
        seed=args.seed,
    )
    model = HymnPlus(cfg)
    print(f"HYMN-Plus trainable params: {count_params(model):,}")

    result = train(
        model,
        corpus,
        char_to_id,
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
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, **sd)
    meta = {
        "arch": "hymn_plus_v1",
        "dim": args.dim,
        "n_layers": args.n_layers,
        "mlp_mult": args.mlp_mult,
        "vocab_size": cfg.vocab_size,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "warm_start_chars": args.warm_start_chars,
        "trainable_params": count_params(model),
        "wall_seconds": result["wall_seconds"],
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "device": _device_name(dev),
        "seed": args.seed,
        "char_vocab": sorted(char_to_id.keys()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {out} + {out.with_suffix('.json')}")
    print(
        f"steps={result['steps']}  wall={result['wall_seconds']:.1f}s  "
        f"initial_loss={result['initial_loss']:.4f}  final_loss={result['final_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
