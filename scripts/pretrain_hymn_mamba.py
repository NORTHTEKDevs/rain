"""Training driver for HYMN-Mamba (HYMN-Plus stack with proper Mamba S6 block).

A/B comparison spec: use the SAME --dim, --n-layers, --steps, --lr,
etc. as a HYMN-Plus v1 run, change only --d-state / --expand to control
S6 capacity. If HYMN-Mamba meaningfully beats HYMN-Plus at the same
budget, the simplified recurrence WAS the bottleneck.
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

from rain.core.hymn_mamba import HymnMamba, HymnMambaConfig, count_params


def _build_char_vocab(text: str) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(set(text)))}


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
    total_tokens = 0
    for _ in range(0, n_windows, batch_size):
        starts = rng.integers(0, len(val_ids) - seq_len - 1, size=batch_size)
        windows = np.stack([val_ids[s : s + seq_len + 1] for s in starts])
        x = torch.as_tensor(windows[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(windows[:, 1:], device=device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += y.numel()
    model.train(was_training)
    return total_loss / max(1, total_tokens)


def train(model, train_ids, val_ids, **kwargs) -> dict:
    n_steps = kwargs["n_steps"]
    batch_size = kwargs["batch_size"]
    seq_len = kwargs["seq_len"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    warmup_steps = kwargs["warmup_steps"]
    cosine_decay = kwargs["cosine_decay"]
    grad_clip = kwargs["grad_clip"]
    device = kwargs["device"]
    seed = kwargs["seed"]
    log_every = kwargs["log_every"]
    val_every = kwargs["val_every"]
    early_stop_patience = kwargs["early_stop_patience"]

    model.to(device)
    model.train()

    # AdamW with no-weight-decay on norms/embeddings (modern standard)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Mamba parameters with _no_weight_decay flag (A_log, D, dt bias)
        if (
            getattr(p, "_no_weight_decay", False)
            or p.ndim < 2
            or "norm" in n.lower()
            or "embed" in n.lower()
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )

    rng = np.random.default_rng(seed)
    N = len(train_ids)
    losses: list[float] = []
    val_hist: list[tuple[int, float]] = []
    initial_loss = None
    best_val = float("inf")
    n_no_imp = 0
    stopped_early = None
    start = time.time()

    for step in range(n_steps):
        cur_lr = _lr_schedule(step, n_steps, lr, warmup_steps, cosine_decay)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        starts = rng.integers(0, N - seq_len - 1, size=batch_size)
        batch = np.stack([train_ids[s : s + seq_len + 1] for s in starts])
        x = torch.as_tensor(batch[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(batch[:, 1:], device=device, dtype=torch.long)
        logits = model(x)
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
            avg = sum(losses[-log_every:]) / log_every
            msg = (
                f"step {step + 1}/{n_steps}  "
                f"train(avg last {log_every}, nats/tok) = {avg:.4f}  lr={cur_lr:.2e}"
            )
            if val_ids is not None and (step + 1) % val_every == 0:
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
    p = argparse.ArgumentParser(description="HYMN-Mamba pretrain")
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=800)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    print(f"device: {dev}")

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    char_to_id = _build_char_vocab(corpus)
    print(f"corpus: {len(corpus):,} chars, {len(char_to_id)} unique")
    ids = np.fromiter((char_to_id[c] for c in corpus if c in char_to_id), dtype=np.int64)

    if args.val_split > 0:
        n_val = max(args.seq_len * 8, int(len(ids) * args.val_split))
        n_val = min(n_val, len(ids) // 4)
        train_ids = ids[:-n_val]
        val_ids = ids[-n_val:]
        print(f"split: train={len(train_ids):,} val={len(val_ids):,}")
    else:
        train_ids = ids
        val_ids = None

    cfg = HymnMambaConfig(
        vocab_size=len(char_to_id),
        dim=args.dim,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        seed=args.seed,
    )
    model = HymnMamba(cfg)
    print(f"HYMN-Mamba trainable params: {count_params(model):,}")

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, **sd)
    meta = {
        "arch": "hymn_mamba_v1",
        "dim": args.dim,
        "n_layers": args.n_layers,
        "mlp_mult": args.mlp_mult,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "vocab_size": cfg.vocab_size,
        "dropout": args.dropout,
        "steps": args.steps,
        "stopped_early_at": result.get("stopped_early_at"),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "cosine_decay": args.cosine_decay,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "trainable_params": count_params(model),
        "wall_seconds": result["wall_seconds"],
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "final_val_loss": result.get("final_val_loss"),
        "best_val_loss": result.get("best_val_loss"),
        "val_history": result.get("val_history", []),
        "device": str(dev),
        "seed": args.seed,
        "char_vocab": sorted(char_to_id.keys()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    best_str = (
        f"  best_val={result['best_val_loss']:.4f}"
        if result.get("best_val_loss") is not None
        else ""
    )
    print(f"saved {out}")
    print(
        f"steps={result['steps']}  wall={result['wall_seconds']:.1f}s  final_train={result['final_loss']:.4f}{best_str}"
    )


if __name__ == "__main__":
    main()
