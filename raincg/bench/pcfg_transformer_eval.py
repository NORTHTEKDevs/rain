"""Evaluate a trained transformer checkpoint on PCFG pairs (exact-match)."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

from raincg.bench.common import ExactMatchResult, exact_match
from raincg.bench.transformer_baseline import load_checkpoint, greedy_decode, param_count

REPO = Path(__file__).resolve().parents[2]
PCFG_DIR = REPO / "data" / "pcfg"


@dataclass
class TransformerRun:
    result: ExactMatchResult
    eval_seconds: float
    params: int


def eval_transformer_on_pairs(ckpt_path, pairs, max_output_len=800, device="cpu") -> TransformerRun:
    model, vocab = load_checkpoint(ckpt_path, device=device)
    preds: list[list[str]] = []
    golds: list[list[str]] = []
    t0 = time.perf_counter()
    for src, tgt in pairs:
        preds.append(greedy_decode(model, vocab, src, max_len=max_output_len + 2))
        golds.append(tgt)
    return TransformerRun(result=exact_match(preds, golds),
                          eval_seconds=time.perf_counter() - t0,
                          params=param_count(model))
