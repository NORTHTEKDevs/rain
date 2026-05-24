# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Sample text from a HYMN-Plus v2 checkpoint, optionally swapping in a
fresh KB at inference time.

Demonstrates v2's killer feature: provide a different --kb-source and
the model generates differently WITHOUT any retraining. World knowledge
lives in the KB buffer, not in the weights.

Usage:
    python -m scripts.sample_hymn_plus_v2 \
        --checkpoint data/checkpoints/hymn_plus_v2_bpe.npz \
        --prompt "ROMEO: " --max-new 200

    # Swap in a KB derived from the agent's seeded ShardedKB:
    python -m scripts.sample_hymn_plus_v2 \
        --checkpoint data/checkpoints/hymn_plus_v2_bpe.npz \
        --kb-source data/kb_seed/llama3b_expanded_filtered.jsonl \
        --prompt "ROMEO: " --max-new 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config
from rain.tokenize.bpe import BPETokenizer


def _load_v2(ckpt_path: Path) -> tuple[HymnPlusV2, BPETokenizer, dict]:
    meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("arch") != "hymn_plus_v2":
        raise ValueError(f"checkpoint arch={meta.get('arch')!r}; this loader is for hymn_plus_v2")

    cfg = HymnPlusV2Config(
        vocab_size=meta["vocab_size"],
        dim=meta["dim"],
        n_layers=meta["n_layers"],
        mlp_mult=meta["mlp_mult"],
        kb_size=meta["kb_size"],
        kb_top_k=meta["kb_top_k"],
        kb_attn_in_layers=(
            tuple(meta["kb_attn_in_layers"]) if meta.get("kb_attn_in_layers") else None
        ),
        tie_weights=meta.get("tie_weights", True),
        seed=meta.get("seed", 42),
    )
    model = HymnPlusV2(cfg)
    sd = dict(np.load(ckpt_path))
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    model.eval()

    bpe_path = meta.get("bpe_model_path") or str(ckpt_path.with_suffix(".bpe.model"))
    tok = BPETokenizer(vocab_size=meta.get("bpe_vocab", cfg.vocab_size))
    tok.load(bpe_path)
    return model, tok, meta


def _kb_from_jsonl(jsonl_path: Path, dim: int, kb_size: int, seed: int = 0) -> np.ndarray:
    """Build a (kb_size, dim) bipolar KB matrix from a fact JSONL.

    Each fact's hypervector is constructed by binding (subject, relation,
    object) via the RAIN codebook + bind / bundle primitives. Pads with
    bipolar-random vectors if there are fewer facts than kb_size; truncates
    if there are more.
    """
    from rain.core.relational import Codebook, bind, bundle

    cb = Codebook(vocab_size=8192, dim=dim, seed=seed)
    rng = np.random.default_rng(seed + 1)

    rows: list[np.ndarray] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = rec.get("subject") or rec.get("s")
            r = rec.get("relation") or rec.get("r")
            o = rec.get("object") or rec.get("o")
            if not (s and r and o):
                continue
            sv = cb.vector(str(s))
            rv = cb.vector(str(r))
            ov = cb.vector(str(o))
            fact_hv = bundle([bind(sv, rv), ov])
            rows.append(fact_hv.astype(np.float32))
            if len(rows) >= kb_size:
                break

    if len(rows) < kb_size:
        pad = kb_size - len(rows)
        pad_rows = (rng.integers(0, 2, size=(pad, dim)) * 2 - 1).astype(np.float32)
        if rows:
            rows = rows + list(pad_rows)
        else:
            rows = list(pad_rows)

    return np.stack(rows[:kb_size], axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="ROMEO: ")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument(
        "--kb-source",
        default=None,
        help="optional KB JSONL whose facts will be hashed into the model's KB "
        "buffer before generation (demonstrates tell-changes-generation without retraining)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--show-kb-attn",
        action="store_true",
        help="print the top facts attended to at each generated token",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = Path(args.checkpoint)
    model, tok, meta = _load_v2(ckpt)
    print(
        f"loaded {ckpt.name} (dim={meta['dim']} layers={meta['n_layers']} "
        f"kb_size={meta['kb_size']} bpe_vocab={meta['vocab_size']})"
    )

    if args.kb_source:
        kb_np = _kb_from_jsonl(Path(args.kb_source), dim=meta["dim"], kb_size=meta["kb_size"])
        model.set_kb(torch.as_tensor(kb_np, dtype=torch.float32))
        print(f"swapped KB from {args.kb_source} ({kb_np.shape})")

    ids = tok.encode(args.prompt)
    if not ids:
        ids = [0]
    prompt = torch.tensor([ids], dtype=torch.long)
    out = model.sample(prompt, max_new=args.max_new, temperature=args.temperature, top_k=args.top_k)
    full_ids = out[0].tolist()
    text = tok.decode(full_ids)
    print()
    # Force utf-8 for Windows cp1252 stdout
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
