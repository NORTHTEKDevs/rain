"""Train HYMN-Mini (Phase 1b) -- the three-way fusion.

8x8 toroidal field, D=1000, ~12.5M params. Curriculum:
  steps 0 to phase1_steps: TBPTT with 5 unrolled steps.
  steps phase1_steps to end: DEQ with IFT + Anderson in soft-tanh space.

This script runs against Tiny Shakespeare for parity with the substrate
experiment. For SCAN/COGS evaluation, see train_hymn_scan.py (TODO; Phase 1b
secondary target).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from hymn import HYMNConfig, HYMNMini
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent.resolve()


class CharDataset(Dataset):
    def __init__(self, ids: np.ndarray, seq_len: int) -> None:
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ids) - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.ids[idx : idx + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(seq[:-1]), torch.from_numpy(seq[1:])


def evaluate(model: HYMNMini, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1_000)
    p.add_argument("--hidden", type=int, default=2_048)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)        # smaller; full field per step
    p.add_argument("--seq", type=int, default=64)          # short seq for HYMN-Mini
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--phase1-steps", type=int, default=50_000)
    p.add_argument("--phase1-unroll", type=int, default=5)
    p.add_argument("--phase2-max-iter", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--frozen-codebook", action="store_true",
                   help="Force codebook frozen (audit-strict v1.1).")
    p.add_argument("--readout-zone-only", action="store_true",
                   help="Read only from output zone (audit-strict v1.1).")
    p.add_argument("--anneal-steps", type=int, default=50_000)
    p.add_argument("--zone-anneal-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(HERE / "checkpoints" / "hymn_mini.pt"))
    args = p.parse_args()

    device = torch.device(args.device)
    data_dir = HERE / "data" / "tinyshakespeare"
    train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint8)
    val_ids = np.fromfile(data_dir / "val.bin", dtype=np.uint8)

    train_ds = CharDataset(train_ids, args.seq)
    val_ds = CharDataset(val_ids, args.seq)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    cfg = HYMNConfig(
        height=args.height, width=args.width, d=args.d,
        vocab_size=65, hidden=args.hidden,
        phase1_steps=args.phase1_steps,
        phase1_unroll=args.phase1_unroll,
        phase2_max_iter=args.phase2_max_iter,
        beta_anneal_steps=args.anneal_steps,
        zone_loss_anneal_steps=args.zone_anneal_steps,
        learnable_codebook=not args.frozen_codebook,
        readout_all_cells=not args.readout_zone_only,
    )
    model = HYMNMini(cfg).to(device)
    print(f"HYMN-Mini params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    t0 = time.time()
    best_val = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            _, loss, metrics = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.step_anneal()

            if step % args.log_every == 0:
                dt = time.time() - t0
                phase = "P1-TBPTT" if step < args.phase1_steps else "P2-DEQ"
                msg = (
                    f"step {step:6d} [{phase}] loss {loss.item():.4f} "
                    f"(ce {metrics.get('ce_loss', float('nan')):.4f}, "
                    f"zone {metrics.get('zone_loss', 0.0):.4f}) "
                    f"beta {metrics['beta']:.2f} | {dt:.1f}s"
                )
                print(msg)

            if step > 0 and step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                bpc = val_loss / math.log(2)
                print(f"  val_loss {val_loss:.4f} (bpc {bpc:.3f})")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {"model": model.state_dict(), "cfg": cfg.__dict__,
                         "step": step, "val_loss": val_loss},
                        args.out,
                    )

            step += 1

    val_loss = evaluate(model, val_loader, device, max_batches=100)
    bpc = val_loss / math.log(2)
    print("=" * 60)
    print(f"HYMN-Mini FINAL val loss {val_loss:.4f} (bpc {bpc:.3f})")


if __name__ == "__main__":
    main()
