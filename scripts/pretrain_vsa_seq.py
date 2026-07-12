"""Training driver for the VSA-Seq engine.

Mirrors `scripts/pretrain_hymn_torch.py` so the A/B with HYMN is
apples-to-apples (same corpus, same batch size, same carry-step
sequence training, same AdamW + warmup + cosine schedule).

Usage:
    python -m scripts.pretrain_vsa_seq \
        --corpus data/corpora/tiny_shakespeare.txt \
        --steps 50000 --batch-size 16 --carry-steps 16 \
        --dim 1024 --lr 5e-4 --warmup-steps 500 --cosine-decay \
        --grad-clip 1.0 --warm-start-chars \
        --device directml \
        --out data/checkpoints/vsa_seq_v1_50k.npz
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

from rain.core.relational import Codebook
from rain.core.vsa_seq import VSASeq, VSASeqConfig, count_params
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


def _build_codebook_from_rain(rain_cb: Codebook, char_to_id: dict[str, int]) -> np.ndarray:
    """Materialize a (V, D) bipolar numpy matrix from RAIN's lazy Codebook,
    in the order char_to_id expects.
    """
    rows = []
    for ch, _ in sorted(char_to_id.items(), key=lambda kv: kv[1]):
        rows.append(rain_cb.vector(ch))
    return np.stack(rows, axis=0).astype(np.int16)


def _lr_schedule(step: int, total: int, peak_lr: float, warmup: int, cosine: bool) -> float:
    """Linear warmup then optional cosine decay to 10% of peak."""
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    if not cosine:
        return peak_lr
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (0.1 + 0.9 * cos)


def train(
    model: VSASeq,
    corpus: str,
    char_to_id: dict[str, int],
    *,
    n_steps: int,
    batch_size: int,
    carry_steps: int,
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

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    ids = np.fromiter((char_to_id[c] for c in corpus if c in char_to_id), dtype=np.int64)
    N = len(ids)
    span = max(2, carry_steps + 1)

    losses: list[float] = []
    initial_loss = None
    start = time.time()

    for step in range(n_steps):
        # Set LR for this step
        cur_lr = _lr_schedule(step, n_steps, lr, warmup_steps, cosine_decay)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Sample batch: (B, span) windows. Model.forward returns logits[t]
        # predicting tokens[t] (BEFORE bundling token[t] into state), so we
        # pass the full window as input AND target.
        starts = rng.integers(0, N - span - 1, size=batch_size)
        batch = np.stack([ids[s : s + span] for s in starts], axis=0)
        x = torch.as_tensor(batch, device=device, dtype=torch.long)  # (B, T)

        logits, _ = model(x)
        # CE: logits[t] vs x[t]; the model never sees x[t] before predicting it.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x.reshape(-1),
            reduction="mean",
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

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
    return {
        "steps": n_steps,
        "wall_seconds": wall,
        "initial_loss": initial_loss,
        "final_loss": float(sum(losses[-min(2000, len(losses)) :]) / min(2000, len(losses))),
        "losses": losses,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="VSA-Seq pretrain (PyTorch + DirectML)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--carry-steps", type=int, default=16)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-perm-lanes", type=int, default=4)
    p.add_argument("--bipolar-sharpness", type=float, default=1.0)
    p.add_argument("--warm-start-chars", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "directml"], default="auto")
    p.add_argument("--log-every", type=int, default=1000)
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

    # Build codebook (warm-started or random)
    rain_cb = Codebook(vocab_size=256, dim=args.dim, seed=args.seed)
    if args.warm_start_chars:
        ws = warm_start_chars(rain_cb, corpus, seed=args.seed)
        print(f"warm-start[features]: {ws}")
    cb_np = _build_codebook_from_rain(rain_cb, char_to_id)

    cfg = VSASeqConfig(
        vocab_size=len(char_to_id),
        dim=args.dim,
        n_perm_lanes=args.n_perm_lanes,
        bipolar_sharpness=args.bipolar_sharpness,
        seed=args.seed,
    )
    model = VSASeq(cfg, codebook=cb_np)
    print(f"VSA-Seq trainable params: {count_params(model):,}")

    result = train(
        model,
        corpus,
        char_to_id,
        n_steps=args.steps,
        batch_size=args.batch_size,
        carry_steps=args.carry_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cosine_decay=args.cosine_decay,
        grad_clip=args.grad_clip,
        device=dev,
        seed=args.seed,
        log_every=args.log_every,
    )

    # Save checkpoint (npz of state_dict tensors + JSON meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, **sd)
    meta = {
        "arch": "vsa_seq_v1",
        "dim": args.dim,
        "vocab_size": cfg.vocab_size,
        "n_perm_lanes": cfg.n_perm_lanes,
        "bipolar_sharpness": cfg.bipolar_sharpness,
        "steps": args.steps,
        "carry_steps": args.carry_steps,
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
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {out} + {meta_path}")
    print(
        f"steps={result['steps']}  carry={args.carry_steps}  "
        f"wall={result['wall_seconds']:.1f}s  "
        f"initial_loss={result['initial_loss']:.4f}  "
        f"final_loss={result['final_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
