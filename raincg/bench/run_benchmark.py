"""One-command PCFG head-to-head: VSA vs Transformer.

Usage:
  python -m raincg.bench.run_benchmark --all                 # full benchmark
  python -m raincg.bench.run_benchmark --all --quick         # tiny, fast smoke
  python -m raincg.bench.run_benchmark --all --max-tgt-len 150  # length-bounded
  python -m raincg.bench.run_benchmark --falsify             # VSA sanity (~0%)

Notes:
- --quick subsets BOTH train (2000) and test (300) so the harness completes
  fast; it is a harness check, NOT the headline numbers.
- --max-tgt-len N filters train AND test to examples whose target length <= N,
  applied identically to both systems (an honest, documented length bound).
- The transformer decode budget is tied to the length bound so greedy decode
  stops early instead of running the full 800-step budget per example.
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pure_vsa.pcfg_hyperion import load_pcfg_split
from raincg.bench.common import ExactMatchResult, wilson_ci
from raincg.bench.pcfg_vsa_eval import run_vsa_pcfg
from raincg.bench.transformer_baseline import train_baseline
from raincg.bench.pcfg_transformer_eval import eval_transformer_on_pairs

REPO = Path(__file__).resolve().parents[2]
PCFG_DIR = REPO / "data" / "pcfg"
RESULTS_DIR = REPO / "raincg" / "results"


@dataclass
class BenchRow:
    name: str
    params: int
    train_time: str
    eval_time: str
    accuracy: float
    correct: int
    total: int


def render_table(rows: list[BenchRow]) -> str:
    head = (f"{'System':<22}{'Params':>12}{'Train':>10}{'Eval':>10}"
            f"{'Accuracy':>12}{'Correct':>14}")
    lines = [head, "-" * len(head)]
    for r in rows:
        correct_str = f"{r.correct}/{r.total}"
        lines.append(f"{r.name:<22}{r.params:>12,}{r.train_time:>10}{r.eval_time:>10}"
                     f"{r.accuracy:>11.2%} {correct_str:>14}")
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 90 else f"{seconds / 60:.1f}min"


def _filter_by_tgt_len(pairs, max_tgt_len):
    if max_tgt_len is None:
        return pairs
    return [(s, t) for s, t in pairs if len(t) <= max_tgt_len]


def _seeded_subsample(pairs, k, seed):
    """Deterministic random subsample of k pairs, order-preserving.
    Returns pairs unchanged if k is None or >= len(pairs)."""
    if k is None or k >= len(pairs):
        return pairs
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(pairs)), k))
    return [pairs[i] for i in idx]


def write_result_json(out_dir: Path, *, bench: str, system: str, split: str,
                      result: ExactMatchResult, params: int, train_s: float,
                      eval_s: float, seed: int, config: dict,
                      notes: str = "") -> Path:
    """Write a single result JSON per the shared results-contract schema."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = wilson_ci(result.correct, result.total)
    payload = {
        "bench": bench,
        "system": system,
        "split": split,
        "n": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "ci95": [lo, hi],
        "params": params,
        "train_s": train_s,
        "eval_s": eval_s,
        "seed": seed,
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_tier": "measured-fresh",
        "notes": notes,
    }
    path = out_dir / f"{bench}__{system}__seed{seed}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run VSA + Transformer")
    ap.add_argument("--quick", action="store_true", help="tiny subset, fast smoke")
    ap.add_argument("--falsify", action="store_true", help="VSA sanity check only")
    ap.add_argument("--vsa-d", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--max-tgt-len", type=int, default=None,
                    help="filter train+test to target length <= N (both systems)")
    ap.add_argument("--train-limit", type=int, default=None,
                    help="cap transformer train-set size (document if used)")
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt",
                    default=str(REPO / "raincg_pcfg_baseline.pt"),
                    help="transformer checkpoint path")
    ap.add_argument("--resume", action="store_true",
                    help="resume transformer training from --ckpt if present")
    ap.add_argument("--test-limit", type=int, default=None,
                    help="seeded random subsample of test pairs, paired across "
                         "vsa+transformer (overrides --quick's test slice)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --test-limit subsample and result records")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="directory to write contract-schema result JSONs")
    a = ap.parse_args()

    quick_test_limit = 300 if a.quick else None
    train_limit = a.train_limit if a.train_limit is not None else (2000 if a.quick else None)
    vsa_d = 4096 if a.quick else a.vsa_d
    out_dir = Path(a.out_dir)
    bench_name = "pcfg_set"
    split = f"test_tgt_le_{a.max_tgt_len}" if a.max_tgt_len is not None else "test"

    if a.falsify:
        res = run_vsa_pcfg(d=vsa_d, limit=quick_test_limit, falsify=True)
        print(f"FALSIFY VSA: {res.result.correct}/{res.result.total} "
              f"= {res.result.accuracy:.4%} (expect ~0%)")
        return

    # Decode budget: tie to length bound when set, else PCFG max (736) + slack.
    decode_cap = (a.max_tgt_len + 2) if a.max_tgt_len is not None else 760

    rows: list[BenchRow] = []

    if a.all:
        # Load once and apply the SAME filtering to both systems.
        train_all = load_pcfg_split(PCFG_DIR / "train.src", PCFG_DIR / "train.tgt")
        test_all = load_pcfg_split(PCFG_DIR / "test.src", PCFG_DIR / "test.tgt")
        train = _filter_by_tgt_len(train_all, a.max_tgt_len)
        test = _filter_by_tgt_len(test_all, a.max_tgt_len)
        if a.max_tgt_len is not None:
            print(f"[note] length bound tgt<={a.max_tgt_len}: "
                  f"train {len(train)}/{len(train_all)}, test {len(test)}/{len(test_all)}")
        if a.test_limit is not None:
            test = _seeded_subsample(test, a.test_limit, a.seed)
            print(f"[note] test-limit={a.test_limit} seed={a.seed}: "
                  f"paired subsample of {len(test)} examples")
        elif quick_test_limit is not None:
            test = test[:quick_test_limit]
        if train_limit is not None:
            train = train[:train_limit]
            print(f"[note] transformer train-set capped at {len(train)} examples")

        # VSA on the SAME test set (filtered + limited).
        vsa = run_vsa_pcfg(d=vsa_d, limit=None, falsify=False,
                           test_override=test)
        rows.append(BenchRow("VSA (pure_vsa)", 0, _fmt(vsa.fit_seconds),
                             _fmt(vsa.eval_seconds), vsa.result.accuracy,
                             vsa.result.correct, vsa.result.total))
        write_result_json(out_dir, bench=bench_name, system="pure_vsa", split=split,
                          result=vsa.result, params=0, train_s=vsa.fit_seconds,
                          eval_s=vsa.eval_seconds, seed=a.seed,
                          config=dict(vsa_d=vsa_d, max_tgt_len=a.max_tgt_len,
                                      test_limit=a.test_limit, quick=a.quick))

        d_model = 64 if a.quick else 512
        nlayers = 1 if a.quick else 2
        ff = 128 if a.quick else 2048
        nhead = 2 if a.quick else 4
        print(f"[train] transformer d_model={d_model} layers={nlayers} ff={ff} "
              f"epochs={a.epochs} max_minutes={a.max_minutes} on {len(train)} examples",
              flush=True)
        tr = train_baseline(train, d_model=d_model, nhead=nhead,
                            num_layers=nlayers, ff=ff, epochs=a.epochs,
                            ckpt_path=a.ckpt, max_minutes=a.max_minutes,
                            seed=a.seed, resume=a.resume)
        te = eval_transformer_on_pairs(a.ckpt, test, max_output_len=decode_cap)
        rows.append(BenchRow("Transformer", te.params, _fmt(tr.train_seconds),
                             _fmt(te.eval_seconds), te.result.accuracy,
                             te.result.correct, te.result.total))
        write_result_json(out_dir, bench=bench_name,
                          system=f"transformer_d{d_model}x{nlayers}", split=split,
                          result=te.result, params=te.params, train_s=tr.train_seconds,
                          eval_s=te.eval_seconds, seed=a.seed,
                          config=dict(d_model=d_model, nhead=nhead, num_layers=nlayers,
                                      ff=ff, epochs=a.epochs, max_minutes=a.max_minutes,
                                      max_tgt_len=a.max_tgt_len, test_limit=a.test_limit,
                                      train_limit=train_limit, quick=a.quick,
                                      resume=a.resume, ckpt=a.ckpt))
    else:
        vsa = run_vsa_pcfg(d=vsa_d, limit=quick_test_limit, falsify=False)
        rows.append(BenchRow("VSA (pure_vsa)", 0, _fmt(vsa.fit_seconds),
                             _fmt(vsa.eval_seconds), vsa.result.accuracy,
                             vsa.result.correct, vsa.result.total))
        write_result_json(out_dir, bench=bench_name, system="pure_vsa", split=split,
                          result=vsa.result, params=0, train_s=vsa.fit_seconds,
                          eval_s=vsa.eval_seconds, seed=a.seed,
                          config=dict(vsa_d=vsa_d, max_tgt_len=a.max_tgt_len,
                                      quick=a.quick))

    print()
    print(render_table(rows))
    if len(rows) == 2:
        gap = rows[0].accuracy - rows[1].accuracy
        print(f"\nGap (VSA - Transformer): {gap:+.2%}")
        passed = rows[0].accuracy >= 0.99 and rows[1].accuracy < 0.95
        print("GATE:", "PASS - proceed to Phase 2" if passed
              else "HALT - gate not met; do not enter Phase 2")


if __name__ == "__main__":
    main()
