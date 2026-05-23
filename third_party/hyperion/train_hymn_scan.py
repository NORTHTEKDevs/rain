"""Train HYMN-Mini on SCAN compositional-generalization splits.

Phase 1b experiment. Default target: add-primitive jump split.
  Transformer baseline: ~2%.
  HYMN target: >60% exact-match.

Usage:
  python data/scan/download_and_prep.py     # one-time
  python train_hymn_scan.py --split addprim_jump
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from eval_scan import evaluate_split
from hymn import HYMNConfig, HYMNMini
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent.resolve()


class ScanDataset(Dataset):
    """Each item is one full <BOS> ... <SEP> ... <EOS> sequence padded to max_len.

    For training, the (input, target) pair is the standard LM shift-by-one,
    with a loss mask zeroing out <PAD> positions.
    """

    def __init__(self, arr: np.ndarray, pad_id: int) -> None:
        self.arr = arr
        self.pad_id = pad_id

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.arr[idx].astype(np.int64)
        x = torch.from_numpy(seq[:-1])
        y = torch.from_numpy(seq[1:])
        mask = (y != self.pad_id).to(torch.float32)
        return x, y, mask


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on (B, T, V) logits with a (B, T) float mask of valid positions."""
    V = logits.size(-1)
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none",
    ).reshape_as(targets)  # (B, T)
    return (ce * mask).sum() / mask.sum().clamp(min=1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="addprim_jump",
                   choices=["simple", "addprim_jump", "length"])
    p.add_argument("--d", type=int, default=1_000)
    p.add_argument("--hidden", type=int, default=2_048)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--n-input", type=int, default=16)
    p.add_argument("--n-comp", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--phase1-steps", type=int, default=50_000)
    p.add_argument("--phase1-unroll", type=int, default=5)
    p.add_argument("--phase2-max-iter", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--learnable-codebook", action="store_true", default=True,
                   help="Codebook is learnable. Pilot showed this is required at sub-1M params.")
    p.add_argument("--frozen-codebook", action="store_true",
                   help="Force codebook frozen (audit-strict v1.1). Disables --learnable-codebook.")
    p.add_argument("--readout-zone-only", action="store_true",
                   help="Read only from output zone (audit-strict v1.1). "
                        "Pilot showed this caps learning at small scale.")
    p.add_argument("--anneal-steps", type=int, default=50_000)
    p.add_argument("--zone-anneal-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-n", type=int, default=200,
                   help="Periodic-eval sample size.")
    p.add_argument("--final-eval-n", type=int, default=2_000,
                   help="Final-eval sample size. Full SCAN test set is 7K-15K "
                        "depending on split; capping prevents multi-hour CPU hangs.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    split_dir = HERE / "data" / "scan" / args.split
    meta = json.loads((split_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = meta["vocab_size"]
    pad_id = meta["pad_id"]
    seq_len = meta["max_len"]
    device = torch.device(args.device)
    print(f"SCAN/{args.split}: vocab={vocab_size}, max_len={seq_len}, "
          f"train={meta['n_train']}, test={meta['n_test']}")

    train_arr = np.fromfile(split_dir / "train.bin", dtype=np.uint16).reshape(-1, seq_len)
    train_ds = ScanDataset(train_arr, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    learnable_codebook = args.learnable_codebook and not args.frozen_codebook
    readout_all_cells = not args.readout_zone_only
    cfg = HYMNConfig(
        height=args.height, width=args.width, d=args.d,
        n_input=args.n_input, n_comp=args.n_comp,
        vocab_size=vocab_size, hidden=args.hidden,
        phase1_steps=args.phase1_steps,
        phase1_unroll=args.phase1_unroll,
        phase2_max_iter=args.phase2_max_iter,
        beta_anneal_steps=args.anneal_steps,
        zone_loss_anneal_steps=args.zone_anneal_steps,
        learnable_codebook=learnable_codebook,
        readout_all_cells=readout_all_cells,
    )
    print(f"Config: learnable_codebook={learnable_codebook}, readout_all_cells={readout_all_cells}")
    model = HYMNMini(cfg).to(device)
    print(f"HYMN-Mini params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_path = args.out or str(HERE / "checkpoints" / f"hymn_mini_scan_{args.split}.pt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    t0 = time.time()
    step = 0
    model.train()

    while step < args.steps:
        for x, y, mask in train_loader:
            if step >= args.steps:
                break
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _, metrics = model(x, y)
            loss = masked_ce_loss(logits, y, mask)
            # zone aux loss is already included by the model under .forward();
            # here we use the raw masked CE for SCAN because zone targets may be PADs.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.step_anneal()

            if step % args.log_every == 0:
                dt = time.time() - t0
                phase = "P1-TBPTT" if step < args.phase1_steps else "P2-DEQ"
                print(
                    f"step {step:6d} [{phase}] loss {loss.item():.4f} "
                    f"beta {metrics['beta']:.2f} | {dt:.1f}s"
                )

            if step > 0 and step % args.eval_every == 0:
                results = evaluate_split(
                    model, split_dir, n_eval=args.eval_n, max_new=80, verbose=False,
                )
                acc = results["exact_match"]
                print(f"  eval exact-match: {acc*100:.2f}% ({results['correct']}/{results['n']})")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(
                        {"model": model.state_dict(), "cfg": cfg.__dict__,
                         "step": step, "exact_match": acc},
                        out_path,
                    )
                    print(f"  saved {out_path}")
                model.train()
            step += 1

    # Final eval -- capped by --final-eval-n so we don't hang on huge test sets.
    print("=" * 60)
    print(f"Running final eval on {args.final_eval_n} test examples ...")
    results = evaluate_split(
        model, split_dir, n_eval=args.final_eval_n, max_new=80, verbose=False,
    )
    acc = results["exact_match"]
    print(f"FINAL SCAN/{args.split} exact-match: {acc*100:.2f}% "
          f"({results['correct']}/{results['n']})")
    if args.split == "addprim_jump":
        verdict = "PASS" if acc > 0.60 else ("FAIL" if acc < 0.30 else "BORDERLINE")
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
