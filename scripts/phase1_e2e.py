# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Phase-1 end-to-end driver -- the 'press the button' v0.5 entry point.

Composes the bootstrap orchestrator + the trained-recipe HYMN pretrain
into one CLI command. Sequence:

  1. rain.train.bootstrap.bootstrap_phase1(corpus, KB, warm_start) -> RainBootstrap
  2. HymnTorch init with carry-16 + NLL + AdamW + cosine + warm-start-chars
  3. train_torch(...) for N steps
  4. save_torch_checkpoint with full schema-v2 metadata
  5. evaluate L1 on the produced checkpoint
  6. emit a single JSON 'phase1_bundle.json' summarizing everything

This is the script you point at WikiText-103 (or larger) when you have
compute to spend.

Usage:
    python -m scripts.phase1_e2e \\
        --corpus data/corpora/wikitext103_train.txt \\
        --kb data/kb_seed/llama3b_expanded_filtered.jsonl \\
        --steps 100000 --in-dim 1024 --carry-steps 16 \\
        --device directml --out-dir data/phase1_v05_run/

Defaults aim for the L1-passing recipe (dim=1024, carry=16, lr=5e-4,
warm-start). Override via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rain.core.relational import Codebook
from rain.train.bootstrap import bootstrap_phase1
from rain.train.torch_trainer import (
    LOSS_NLL,
    HymnTorch,
    auto_device,
    device_name,
    save_torch_checkpoint,
    train_torch,
)
from rain.train.warm_start_chars import warm_start_chars


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-1 end-to-end pretrain")
    p.add_argument("--corpus", required=True)
    p.add_argument("--kb", default=None, help="optional KB seed JSONL")
    p.add_argument("--out-dir", required=True,
                   help="dir for checkpoint + bundle + l1 report")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--carry-steps", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--in-dim", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "directml"], default="auto")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--skip-warm-start", action="store_true",
                   help="don't seed char features (default: warm-start ON)")
    p.add_argument("--skip-l1-eval", action="store_true",
                   help="skip running L1 NLL on the produced checkpoint")
    p.add_argument("--l1-corpus", default=None,
                   help="corpus to eval L1 on; defaults to --corpus")
    p.add_argument("--l1-n-eval-chars", type=int, default=5000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "hymn_phase1.npz"
    bundle_path = out_dir / "phase1_bundle.json"

    started = time.time()

    print("Phase-1 end-to-end pretrain")
    print("=" * 60)
    print(f"corpus:   {args.corpus}")
    print(f"kb:       {args.kb or '(none)'}")
    print(f"out_dir:  {out_dir}")
    print(f"recipe:   carry={args.carry_steps}, batch={args.batch_size}, "
          f"lr={args.lr}, wd={args.weight_decay}, grad_clip={args.grad_clip}")
    print(f"warm:     {'OFF' if args.skip_warm_start else 'ON'}")
    print()

    # 1. Bootstrap (Phase 1.1-1.7)
    print("[1] bootstrap_phase1 ...")
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    boot = bootstrap_phase1(
        corpus_texts=[corpus_text],
        kb_jsonl_path=args.kb,
        D=args.in_dim,
        vocab_size=256,
        seed=args.seed,
    )
    print(f"    components ready: {list(boot.summary().keys())}")

    # 2. Device
    if args.device == "cpu":
        import torch
        dev = torch.device("cpu")
    elif args.device == "directml":
        import torch_directml as _dml
        dev = _dml.device(0)
    else:
        dev = auto_device()
    print(f"[2] device: {device_name(dev)}")

    # 3. Codebook + warm-start
    cb = Codebook(vocab_size=256, dim=args.in_dim, seed=args.seed)
    if not args.skip_warm_start:
        ws_stats = warm_start_chars(cb, corpus_text, seed=args.seed)
        print(f"[3] warm-start: {ws_stats}")
    else:
        print("[3] warm-start: SKIPPED")

    # 4. Model
    model = HymnTorch(args.in_dim, args.hidden_dim, args.out_dim,
                      seed=args.seed, device=dev)

    # 5. Train
    print(f"[4] train: {args.steps} steps...")
    train_started = time.time()
    result = train_torch(
        model, cb, corpus_text,
        n_steps=args.steps,
        batch_size=args.batch_size,
        carry_steps=args.carry_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cosine_decay=args.cosine_decay,
        grad_clip=args.grad_clip,
        loss_type=LOSS_NLL,
        device=dev,
        seed=args.seed,
        log_every=args.log_every,
    )
    train_wall = time.time() - train_started
    print(f"    train done: {train_wall:.0f}s "
          f"loss {result.initial_loss:.3f} -> {result.final_loss:.3f}")

    # 6. Save
    save_torch_checkpoint(
        model, ckpt_path,
        n_steps=args.steps, lr=args.lr, seed=args.seed,
        initial_loss=result.initial_loss, final_loss=result.final_loss,
        losses=result.losses,
        loss_type=LOSS_NLL, context_len=0,
        carry_steps=args.carry_steps, batch_size=args.batch_size,
    )
    print(f"[5] saved {ckpt_path}")

    # 7. Optional L1 eval
    l1_result = None
    if not args.skip_l1_eval:
        print("[6] L1 eval ...")
        from evals.tier2_llm_parity.tiny_shakespeare import evaluate
        l1_result = evaluate(
            str(ckpt_path),
            args.l1_corpus or args.corpus,
            n_eval_chars=args.l1_n_eval_chars,
            metric="nll",
        )
        print(f"    L1 NLL: {l1_result['nll_loss']:.4f}  "
              f"pass: {l1_result['pass']}")

    bundle = {
        "phase1_e2e": "v0.5",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S",
                                 time.localtime(started)),
        "wall_seconds": time.time() - started,
        "train_wall_seconds": train_wall,
        "args": vars(args),
        "bootstrap_summary": boot.summary(),
        "train": {
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "n_steps": result.steps,
            "device": result.device,
        },
        "checkpoint_path": str(ckpt_path),
        "l1_result": l1_result,
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"[7] bundle written: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
