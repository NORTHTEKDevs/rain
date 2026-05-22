"""SCAN exact-match accuracy evaluator for HYMN-Mini.

Given a trained HYMN-Mini checkpoint and a SCAN split, prompt the model with
each test example's input portion (up through <SEP>) and generate until <EOS>.
Compute exact-match accuracy on the generated action sequence against gold.

This is the falsifiable Phase 1b metric. Per the v1.1 audit:
  Pass on SCAN add-primitive jump: >60%
  Fail:                            <30%
Transformer baseline (Lake & Baroni 2018): ~2%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymn import HYMNMini, HYMNConfig


HERE = Path(__file__).parent.resolve()


def find_indices(seq: list[int], token: int) -> int:
    """Index of `token` in seq, or -1 if absent."""
    for i, t in enumerate(seq):
        if t == token:
            return i
    return -1


def exact_match(pred_actions: list[int], gold_actions: list[int]) -> bool:
    """Strict sequence equality after stripping pads / eos / bos."""
    return pred_actions == gold_actions


def evaluate_split(
    model: HYMNMini,
    split_dir: Path,
    n_eval: int | None = None,
    max_new: int = 64,
    verbose: bool = False,
) -> dict[str, float]:
    meta = json.loads((split_dir / "meta.json").read_text(encoding="utf-8"))
    pad_id = meta["pad_id"]
    sep_id = meta["sep_id"]
    eos_id = meta["eos_id"]
    seq_len = meta["max_len"]
    vocab = meta["vocab"]
    # inverse vocab for verbose printing
    inv_vocab = {int(v): k for k, v in vocab.items()}

    test_arr = np.fromfile(split_dir / "test.bin", dtype=np.uint16).reshape(-1, seq_len)
    if n_eval is not None:
        test_arr = test_arr[:n_eval]
    n = test_arr.shape[0]
    device = next(model.parameters()).device

    correct = 0
    for i in range(n):
        seq = test_arr[i].tolist()
        # find SEP and EOS positions
        sep_pos = find_indices(seq, sep_id)
        if sep_pos < 0:
            continue
        # prompt = everything up through and including SEP
        prompt = torch.tensor(seq[: sep_pos + 1], dtype=torch.long, device=device)
        # gold output = tokens after SEP, up to first EOS or first PAD
        gold = []
        for t in seq[sep_pos + 1:]:
            if t == eos_id or t == pad_id:
                break
            gold.append(t)

        full = model.generate_from_prompt(
            prompt, max_new=max_new, eos_id=eos_id,
            temperature=1.0, top_k=None,
        )
        full_list = full.tolist()
        # predicted output = everything after the SEP in the generated sequence,
        # excluding any final EOS.
        pred = []
        for t in full_list[sep_pos + 1:]:
            if t == eos_id or t == pad_id:
                break
            pred.append(t)

        ok = exact_match(pred, gold)
        if ok:
            correct += 1
        if verbose and i < 10:
            gold_str = " ".join(inv_vocab.get(t, "?") for t in gold)
            pred_str = " ".join(inv_vocab.get(t, "?") for t in pred)
            print(f"[{i}] {'PASS' if ok else 'FAIL'}")
            print(f"     gold: {gold_str}")
            print(f"     pred: {pred_str}")

    acc = correct / max(1, n)
    return {"exact_match": acc, "n": n, "correct": correct}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to HYMN-Mini checkpoint .pt")
    p.add_argument("--split", default="addprim_jump",
                   choices=["simple", "addprim_jump", "length"])
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = HYMNConfig(**ckpt["cfg"])
    model = HYMNMini(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    split_dir = HERE / "data" / "scan" / args.split
    print(f"Evaluating {args.ckpt} on SCAN {args.split} ({split_dir})")
    results = evaluate_split(
        model, split_dir, n_eval=args.n_eval, max_new=args.max_new,
        verbose=args.verbose,
    )
    print(f"\nExact-match accuracy: {results['correct']}/{results['n']} = {results['exact_match']*100:.2f}%")
    if args.split == "addprim_jump":
        print("  >60% PASS | <30% FAIL  (transformer baseline: ~2%)")


if __name__ == "__main__":
    main()
