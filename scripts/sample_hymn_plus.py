"""Sample text from a trained HYMN-Plus checkpoint.

Usage:
    python -m scripts.sample_hymn_plus \
        --checkpoint data/checkpoints/hymn_plus_v1_5k.npz \
        --prompt "ROMEO: " --max-new 200 --temperature 0.7 --top-k 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rain.core.hymn_plus import HymnPlus, HymnPlusConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="ROMEO: ")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    ckpt = Path(args.checkpoint)
    meta = json.loads(ckpt.with_suffix(".json").read_text())

    char_vocab: list[str] = meta["char_vocab"]
    char_to_id = {c: i for i, c in enumerate(char_vocab)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    cfg = HymnPlusConfig(
        vocab_size=len(char_vocab),
        dim=meta["dim"],
        n_layers=meta["n_layers"],
        mlp_mult=meta["mlp_mult"],
        seed=meta["seed"],
    )
    model = HymnPlus(cfg)

    sd = dict(np.load(ckpt))
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    model.eval()

    prompt_ids = [char_to_id[c] for c in args.prompt if c in char_to_id]
    if not prompt_ids:
        prompt_ids = [0]
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    out = model.sample(prompt, max_new=args.max_new, temperature=args.temperature, top_k=args.top_k)
    text = "".join(id_to_char[int(i)] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
