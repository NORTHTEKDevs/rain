"""Evaluate the verified pure-VSA PCFG solver on the PCFG SET nested test."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

from pure_vsa.pcfg_hyperion import PCFGConfig, PCFGHyperion, load_pcfg_split
from raincg.bench.common import ExactMatchResult, exact_match

REPO = Path(__file__).resolve().parents[2]
PCFG_DIR = REPO / "data" / "pcfg"


@dataclass
class VSARun:
    result: ExactMatchResult
    fit_seconds: float
    eval_seconds: float
    d: int
    params: int = 0  # zero trained parameters, by construction


def run_vsa_pcfg(d: int = 8192, limit: int | None = None, falsify: bool = False,
                 test_override: list | None = None) -> VSARun:
    train = load_pcfg_split(PCFG_DIR / "train.src", PCFG_DIR / "train.tgt")
    if test_override is not None:
        test = test_override
    else:
        test = load_pcfg_split(PCFG_DIR / "test.src", PCFG_DIR / "test.tgt")
        if limit is not None:
            test = test[:limit]

    t0 = time.perf_counter()
    r = PCFGHyperion(PCFGConfig(d=d, seed=0, max_output_len=800))
    r.fit(train)
    fit_s = time.perf_counter() - t0

    preds: list[list[str]] = []
    golds: list[list[str]] = []
    t1 = time.perf_counter()
    for src, tgt in test:
        try:
            pred = r.predict_nested(src)
        except Exception:
            pred = []
        preds.append(pred)
        golds.append(list(reversed(tgt)) if falsify else tgt)
    eval_s = time.perf_counter() - t1

    return VSARun(
        result=exact_match(preds, golds),
        fit_seconds=fit_s,
        eval_seconds=eval_s,
        d=d,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args()
    res = run_vsa_pcfg(d=a.d, limit=a.limit, falsify=a.falsify)
    tag = " (FALSIFY)" if a.falsify else ""
    print(f"VSA PCFG{tag}: {res.result.correct}/{res.result.total} "
          f"= {res.result.accuracy:.4%} | D={res.d} | "
          f"fit={res.fit_seconds:.1f}s eval={res.eval_seconds:.1f}s params=0")
