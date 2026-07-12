"""Overnight CPU transformer baseline for PCFG SET, length-bounded.

Trains a real d_model=512 seq2seq Transformer to convergence (time-capped) on
the tgt<=MAX_TGT subset, evaluates it AND the VSA solver on the SAME bounded
test set (apples-to-apples), and writes everything to a JSON result file so the
outcome survives session loss.

Run (background):
  python -m raincg.bench.run_overnight

Tunables via env or edit constants below.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from pure_vsa.pcfg_hyperion import load_pcfg_split
from raincg.bench.pcfg_vsa_eval import run_vsa_pcfg
from raincg.bench.transformer_baseline import train_baseline
from raincg.bench.pcfg_transformer_eval import eval_transformer_on_pairs

REPO = Path(__file__).resolve().parents[2]
PCFG_DIR = REPO / "data" / "pcfg"
RESULT = REPO / "raincg" / "baseline_overnight_result.json"
CKPT = REPO / "raincg_pcfg_baseline_512.pt"
PROGRESS = REPO / "raincg" / "overnight_progress.txt"

MAX_TGT = 40          # documented length bound (~p99 of data)
D_MODEL = 512
NLAYERS = 2
NHEAD = 4
FF = 2048
EPOCHS = 50
MAX_MINUTES = 480.0   # 8h training cap
VSA_D = 8192


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _filt(pairs):
    return [(s, t) for s, t in pairs if len(t) <= MAX_TGT]


def main():
    PROGRESS.write_text("", encoding="utf-8")
    result = {"status": "running", "config": {
        "max_tgt": MAX_TGT, "d_model": D_MODEL, "layers": NLAYERS,
        "ff": FF, "epochs": EPOCHS, "max_minutes": MAX_MINUTES, "vsa_d": VSA_D}}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    train_all = load_pcfg_split(PCFG_DIR / "train.src", PCFG_DIR / "train.tgt")
    test_all = load_pcfg_split(PCFG_DIR / "test.src", PCFG_DIR / "test.tgt")
    train = _filt(train_all)
    test = _filt(test_all)
    _log(f"length bound tgt<={MAX_TGT}: train {len(train)}/{len(train_all)}, "
         f"test {len(test)}/{len(test_all)}")
    result["data"] = {"train": len(train), "train_all": len(train_all),
                      "test": len(test), "test_all": len(test_all)}

    # VSA on the same bounded test set
    _log("running VSA on bounded test set...")
    vsa = run_vsa_pcfg(d=VSA_D, test_override=test)
    result["vsa"] = {"correct": vsa.result.correct, "total": vsa.result.total,
                     "accuracy": vsa.result.accuracy, "fit_s": vsa.fit_seconds,
                     "eval_s": vsa.eval_seconds, "params": 0}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"VSA: {vsa.result.correct}/{vsa.result.total} = {vsa.result.accuracy:.4%}")

    # Transformer train (time-capped)
    _log(f"training transformer d_model={D_MODEL} layers={NLAYERS} ff={FF} "
         f"epochs={EPOCHS} cap={MAX_MINUTES}min on {len(train)} examples...")
    tr = train_baseline(train, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS,
                        ff=FF, epochs=EPOCHS, ckpt_path=str(CKPT),
                        max_minutes=MAX_MINUTES, log_every=1)
    result["transformer_train"] = {"params": tr.params, "epochs_run": tr.epochs_run,
                                   "final_loss": tr.final_loss,
                                   "train_s": tr.train_seconds}
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"trained {tr.epochs_run} epochs, final_loss={tr.final_loss:.4f}, "
         f"params={tr.params:,}, {tr.train_seconds/60:.1f}min")

    # Transformer eval on same bounded test set
    _log("evaluating transformer (greedy decode, this is the slow part)...")
    te = eval_transformer_on_pairs(str(CKPT), test, max_output_len=MAX_TGT + 2)
    result["transformer_eval"] = {"correct": te.result.correct,
                                  "total": te.result.total,
                                  "accuracy": te.result.accuracy,
                                  "eval_s": te.eval_seconds, "params": te.params}
    gap = vsa.result.accuracy - te.result.accuracy
    result["gap"] = gap
    result["gate_pass"] = bool(vsa.result.accuracy >= 0.99 and te.result.accuracy < 0.95)
    result["status"] = "done"
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"transformer: {te.result.correct}/{te.result.total} = {te.result.accuracy:.4%}")
    _log(f"GAP (VSA - Transformer) = {gap:+.2%}  GATE_PASS={result['gate_pass']}")
    _log("DONE")


if __name__ == "__main__":
    main()
