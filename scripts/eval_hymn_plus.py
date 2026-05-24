# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Evaluate a HYMN-Plus checkpoint on a held-out corpus.

Reports NLL (nats/char), perplexity, uniform-baseline ratio, and bits/char.
Strict character-level: skips characters not in the checkpoint's vocab
and reports the coverage ratio so you can tell when the corpus is
out-of-distribution.

Usage:
    python -m scripts.eval_hymn_plus \
        --checkpoint data/checkpoints/hymn_plus_v2_15k.npz \
        --corpus data/corpora/wikitext2_train.txt \
        --n-eval-chars 200000
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


def _load_checkpoint(ckpt_path: Path):
    """Load a HYMN-Plus checkpoint and return (model, char_to_id, meta)."""
    from rain.core.hymn_plus import HymnPlus, HymnPlusConfig

    meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("arch") != "hymn_plus_v1":
        raise ValueError(
            f"checkpoint arch={meta.get('arch')!r}; this evaluator is for hymn_plus_v1"
        )

    char_vocab = meta["char_vocab"]
    char_to_id = {c: i for i, c in enumerate(char_vocab)}
    cfg = HymnPlusConfig(
        vocab_size=len(char_vocab),
        dim=meta["dim"],
        n_layers=meta["n_layers"],
        mlp_mult=meta["mlp_mult"],
        seed=meta.get("seed", 42),
    )
    model = HymnPlus(cfg)
    sd = dict(np.load(ckpt_path))
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    model.eval()
    return model, char_to_id, meta


@torch.no_grad()
def evaluate_nll(
    model,
    char_to_id: dict[str, int],
    text: str,
    *,
    batch_size: int = 16,
    seq_len: int = 256,
    device: str = "cpu",
) -> dict:
    """Stream `text` through the model in (batch_size, seq_len) windows;
    average per-char cross-entropy over the entire text.

    Returns a dict with nll_nats_per_char, perplexity, bits_per_char,
    coverage (fraction of input chars that were in vocab), and timing.
    """
    dev = torch.device(device)
    model.to(dev)

    raw_len = len(text)
    ids_list = [char_to_id[c] for c in text if c in char_to_id]
    coverage = len(ids_list) / max(1, raw_len)
    if len(ids_list) < seq_len + 1:
        return {
            "error": "too few in-vocab chars",
            "coverage": coverage,
            "in_vocab_chars": len(ids_list),
        }

    ids = np.array(ids_list, dtype=np.int64)
    n_windows = (len(ids) - 1) // seq_len
    n_windows = (n_windows // batch_size) * batch_size  # truncate to full batches

    total_loss = 0.0
    total_chars = 0
    start = time.time()
    for b_start in range(0, n_windows, batch_size):
        windows = np.stack(
            [
                ids[i * seq_len : i * seq_len + seq_len + 1]
                for i in range(b_start, b_start + batch_size)
            ]
        )
        x = torch.as_tensor(windows[:, :-1], device=dev, dtype=torch.long)
        y = torch.as_tensor(windows[:, 1:], device=dev, dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_chars += y.numel()

    nll = total_loss / max(1, total_chars)
    return {
        "nll_nats_per_char": nll,
        "perplexity": math.exp(nll),
        "bits_per_char": nll / math.log(2),
        "chars_evaluated": total_chars,
        "coverage": coverage,
        "uniform_baseline_nll": math.log(model.config.vocab_size),
        "ratio_of_uniform": nll / math.log(model.config.vocab_size),
        "wall_seconds": time.time() - start,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Eval a HYMN-Plus checkpoint on held-out text")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument(
        "--n-eval-chars",
        type=int,
        default=0,
        help="cap eval text length (0 = whole corpus)",
    )
    p.add_argument(
        "--offset", type=int, default=0, help="skip the first N chars (e.g. to evaluate a tail)"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"loading {ckpt_path} ...")
    model, char_to_id, meta = _load_checkpoint(ckpt_path)
    print(
        f"  arch={meta['arch']} dim={meta['dim']} layers={meta['n_layers']} vocab={len(char_to_id)}"
    )

    corpus_path = Path(args.corpus)
    text = corpus_path.read_text(encoding="utf-8")
    if args.offset > 0:
        text = text[args.offset :]
    if args.n_eval_chars > 0:
        text = text[: args.n_eval_chars]
    print(f"evaluating on {len(text):,} chars from {corpus_path}")

    result = evaluate_nll(
        model,
        char_to_id,
        text,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )
    result["checkpoint"] = str(ckpt_path)
    result["corpus"] = str(corpus_path)
    result["train_corpus_hint"] = (
        "memorization possible" if len(text) < 5_000_000 else "memorization unlikely"
    )

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
