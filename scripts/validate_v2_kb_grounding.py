# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Architectural validation experiment for HYMN-Plus v2.

The claim: integrating retrieved facts at every block IMPROVES generation
quality when the facts are relevant, vs the same model with a random or
empty KB.

The test: compute held-out NLL on a corpus three times:
  A) v2 with random-init KB (the worst case -- random vectors)
  B) v2 with a KB seeded from the same training corpus (in-distribution
     facts -- the model should benefit)
  C) v2 with a KB seeded from a DIFFERENT corpus (out-of-distribution
     facts -- should be roughly neutral or slightly worse than A,
     showing the model isn't just being randomly perturbed)

If B < A by a meaningful margin (>= 5%), the KB-attention layer is
genuinely USING the KB for prediction. If A == B == C, the layer is
inert (W_o still close to zero) and we have a training problem.

Usage:
    python -m scripts.validate_v2_kb_grounding \
        --checkpoint data/checkpoints/hymn_plus_v2_bpe.npz \
        --train-corpus data/corpora/tiny_shakespeare.txt \
        --eval-corpus data/corpora/tiny_shakespeare.txt \
        --ood-corpus data/corpora/wikitext2_train.txt \
        --n-eval-tokens 5000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config
from rain.tokenize.bpe import BPETokenizer


def _load(ckpt_path: Path) -> tuple[HymnPlusV2, BPETokenizer, dict]:
    meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
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


def _corpus_to_kb(corpus_path: Path, dim: int, kb_size: int, seed: int = 0) -> np.ndarray:
    """Extract noun-phrase-ish 'facts' from a corpus and build a (kb_size, dim)
    bipolar KB matrix.

    We tokenize the corpus into whitespace-separated chunks, take the first
    kb_size unique chunks of >= 4 chars (skipping pure punctuation), and
    encode each as a fact-hypervector via the Codebook.
    """
    from rain.core.relational import Codebook

    text = corpus_path.read_text(encoding="utf-8")
    seen: set[str] = set()
    facts: list[str] = []
    for tok in text.split():
        tok = tok.strip(".,!?:;\"'()[]{}").lower()
        if len(tok) < 4 or not tok.isalpha() or tok in seen:
            continue
        seen.add(tok)
        facts.append(tok)
        if len(facts) >= kb_size:
            break

    cb = Codebook(vocab_size=max(2 * len(facts), 4096), dim=dim, seed=seed)
    rows = [cb.vector(f).astype(np.float32) for f in facts]
    if len(rows) < kb_size:
        rng = np.random.default_rng(seed + 1)
        pad = (rng.integers(0, 2, size=(kb_size - len(rows), dim)) * 2 - 1).astype(np.float32)
        rows.extend(list(pad))
    return np.stack(rows[:kb_size], axis=0)


@torch.no_grad()
def _eval_nll(
    model: HymnPlusV2,
    ids: np.ndarray,
    *,
    batch_size: int = 8,
    seq_len: int = 128,
) -> float:
    model.eval()
    n_windows = (len(ids) - 1) // seq_len
    n_windows = (n_windows // batch_size) * batch_size
    if n_windows == 0:
        return float("nan")
    total_loss = 0.0
    total_tokens = 0
    for b_start in range(0, n_windows, batch_size):
        windows = np.stack(
            [
                ids[i * seq_len : i * seq_len + seq_len + 1]
                for i in range(b_start, b_start + batch_size)
            ]
        )
        x = torch.as_tensor(windows[:, :-1], dtype=torch.long)
        y = torch.as_tensor(windows[:, 1:], dtype=torch.long)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--train-corpus", required=True, help="corpus used to construct the 'in-distribution' KB"
    )
    p.add_argument("--eval-corpus", required=True, help="held-out text to compute NLL on")
    p.add_argument(
        "--ood-corpus", default=None, help="optional second corpus for the OOD KB condition"
    )
    p.add_argument("--n-eval-tokens", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    model, tok, meta = _load(ckpt)
    dim = meta["dim"]
    kb_size = meta["kb_size"]
    print(f"loaded {ckpt.name} (dim={dim} layers={meta['n_layers']} kb_size={kb_size})")

    # Encode eval text
    text = Path(args.eval_corpus).read_text(encoding="utf-8")
    ids = np.array(tok.encode(text), dtype=np.int64)
    if args.n_eval_tokens > 0:
        ids = ids[: args.n_eval_tokens]
    print(f"eval: {len(ids):,} tokens from {args.eval_corpus}")
    print()

    # Original KB (whatever was in the trained checkpoint)
    nll_trained = _eval_nll(model, ids, batch_size=args.batch_size, seq_len=args.seq_len)
    print(
        f"[trained]  KB from checkpoint     -> NLL {nll_trained:.4f}  PPL {math.exp(nll_trained):.2f}"
    )

    # A) Random KB
    rng = np.random.default_rng(99)
    kb_rand = (rng.integers(0, 2, size=(kb_size, dim)) * 2 - 1).astype(np.float32)
    model.set_kb(torch.as_tensor(kb_rand))
    nll_random = _eval_nll(model, ids, batch_size=args.batch_size, seq_len=args.seq_len)
    print(
        f"[A]        random bipolar KB      -> NLL {nll_random:.4f}  PPL {math.exp(nll_random):.2f}"
    )

    # B) In-distribution KB
    kb_id = _corpus_to_kb(Path(args.train_corpus), dim=dim, kb_size=kb_size, seed=7)
    model.set_kb(torch.as_tensor(kb_id))
    nll_id = _eval_nll(model, ids, batch_size=args.batch_size, seq_len=args.seq_len)
    print(f"[B]        in-distribution KB     -> NLL {nll_id:.4f}  PPL {math.exp(nll_id):.2f}")

    # C) OOD KB (optional)
    if args.ood_corpus:
        kb_ood = _corpus_to_kb(Path(args.ood_corpus), dim=dim, kb_size=kb_size, seed=13)
        model.set_kb(torch.as_tensor(kb_ood))
        nll_ood = _eval_nll(model, ids, batch_size=args.batch_size, seq_len=args.seq_len)
        print(
            f"[C]        OOD-corpus KB          -> NLL {nll_ood:.4f}  PPL {math.exp(nll_ood):.2f}"
        )
    else:
        nll_ood = None

    print()
    delta_id = nll_random - nll_id
    pct = delta_id / nll_random * 100 if nll_random > 0 else 0.0
    print(f"verdict: in-distribution KB vs random KB: ΔNLL = {delta_id:+.4f} ({pct:+.2f}%)")
    if delta_id > 0.05 * nll_random:
        print("  -> KB-attention IS using the KB. Architectural moat is functional.")
    elif abs(delta_id) < 0.005 * nll_random:
        print(
            "  -> KB has essentially no effect. KB-attention layer may be inert "
            "(W_o still near zero after training); consider more training or "
            "explicit init to non-zero W_o."
        )
    else:
        print(
            f"  -> Small effect ({pct:+.2f}%). KB-attention is being used but "
            "the in-distribution facts don't help much vs random."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
