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


@torch.no_grad()
def _eval_val(
    model: HymnPlus,
    val_ids: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_windows: int = 64,
) -> float:
    """Compute mean NLL over up to `max_windows` random windows from the
    held-out token stream. Quick (not full-corpus) so it can run every
    N steps without slowing training much.
    """
    was_training = model.training
    model.eval()
    n_avail = (len(val_ids) - 1) // seq_len
    n_windows = min(max_windows, n_avail)
    if n_windows <= 0:
        model.train(was_training)
        return float("nan")
    n_windows = (n_windows // batch_size) * batch_size  # full batches only
    if n_windows == 0:
        n_windows = batch_size
    rng = np.random.default_rng(0)  # deterministic val sample
    total_loss = 0.0
    total_chars = 0
    for b_start in range(0, n_windows, batch_size):
        starts = rng.integers(0, len(val_ids) - seq_len - 1, size=batch_size)
        windows = np.stack([val_ids[s : s + seq_len + 1] for s in starts], axis=0)
        x = torch.as_tensor(windows[:, :-1], device=device, dtype=torch.long)
        y = torch.as_tensor(windows[:, 1:], device=device, dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_chars += y.numel()
    model.train(was_training)
    return total_loss / max(1, total_chars)


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
    val_split: float = 0.0,
    val_every: int = 0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.001,
) -> dict:
    model.to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    all_ids = np.fromiter((char_to_id[c] for c in corpus if c in char_to_id), dtype=np.int64)

    # Split off a contiguous tail for validation (so val never overlaps
    # any window the training loop draws from).
    if val_split > 0.0:
        n_val = max(seq_len * 8, int(len(all_ids) * val_split))
        n_val = min(n_val, len(all_ids) // 4)  # never more than 25%
        train_ids = all_ids[:-n_val]
        val_ids = all_ids[-n_val:]
        print(
            f"split: train={len(train_ids):,} chars, val={len(val_ids):,} chars "
            f"({val_split * 100:.1f}%)"
        )
    else:
        train_ids = all_ids
        val_ids = None
        print(f"no validation split; train={len(train_ids):,} chars (use --val-split to enable)")

    N = len(train_ids)
    losses: list[float] = []
    val_history: list[tuple[int, float]] = []
    initial_loss = None
    start = time.time()

    val_interval = val_every if val_every > 0 else max(1, n_steps // 10)
    best_val = float("inf")
    n_no_improvement = 0
    stopped_early_at: int | None = None

    for step in range(n_steps):
        cur_lr = _lr_schedule(step, n_steps, lr, warmup_steps, cosine_decay)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        starts = rng.integers(0, N - seq_len - 1, size=batch_size)
        batch = np.stack([train_ids[s : s + seq_len + 1] for s in starts], axis=0)
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
            msg = (
                f"step {step + 1}/{n_steps}  "
                f"train(avg last {log_every}) = {avg:.4f}  "
                f"lr={cur_lr:.2e}"
            )
            if val_ids is not None and (step + 1) % val_interval == 0:
                val_nll = _eval_val(
                    model, val_ids, batch_size=batch_size, seq_len=seq_len, device=device
                )
                val_history.append((step + 1, val_nll))
                gap = val_nll - avg
                msg += f"  val = {val_nll:.4f}  gap = {gap:+.4f}"
                if gap > 0.5 and step > warmup_steps + 100:
                    msg += "  WARN: val>>train, likely overfitting"
                # Early stopping on val plateau.
                if val_nll < best_val - early_stop_min_delta:
                    best_val = val_nll
                    n_no_improvement = 0
                else:
                    n_no_improvement += 1
                    if early_stop_patience > 0 and n_no_improvement >= early_stop_patience:
                        msg += (
                            f"  EARLY STOP: val plateaued for {n_no_improvement} "
                            f"checks (patience={early_stop_patience})"
                        )
                        print(msg)
                        stopped_early_at = step + 1
                        break
            print(msg)
        if stopped_early_at is not None:
            break

    wall = time.time() - start
    final = float(sum(losses[-min(1000, len(losses)) :]) / min(1000, len(losses)))
    final_val = val_history[-1][1] if val_history else None
    return {
        "steps": n_steps,
        "stopped_early_at": stopped_early_at,
        "wall_seconds": wall,
        "initial_loss": initial_loss,
        "final_loss": final,
        "final_val_loss": final_val,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "val_history": val_history,
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
    p.add_argument(
        "--no-tie-weights",
        action="store_true",
        help="disable embedding/output weight tying (uses a separate LM head; "
        "+V*D params, slightly more capacity, slightly slower)",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="fraction of corpus held out for validation (default 5%%). "
        "Set to 0 to disable validation.",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="run validation every N steps (0 = ~10 evenly-spaced checks)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop training when val NLL fails to improve for N val checks "
        "(0 = disabled, requires --val-split > 0)",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.001,
        help="minimum val-NLL improvement to reset patience counter",
    )
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
        tie_weights=not args.no_tie_weights,
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
        val_split=args.val_split,
        val_every=args.val_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
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
        "final_val_loss": result.get("final_val_loss"),
        "best_val_loss": result.get("best_val_loss"),
        "stopped_early_at": result.get("stopped_early_at"),
        "val_history": result.get("val_history", []),
        "val_split": args.val_split,
        "early_stop_patience": args.early_stop_patience,
        "device": _device_name(dev),
        "seed": args.seed,
        "char_vocab": sorted(char_to_id.keys()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {out} + {out.with_suffix('.json')}")
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
