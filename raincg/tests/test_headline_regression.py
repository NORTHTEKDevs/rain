"""Pins the headline RAINCG benchmark numbers with real floors, run at full
scale (not mocked, not subsampled). If a real regression lands, these fail
loudly instead of the suite silently going quiet on the numbers we publish.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

RUN_SLOW = os.environ.get("RAINCG_RUN_SLOW_TESTS") == "1"


def test_cogs_template_symbolic_headline_floor():
    """COGS template learner: fit on train.tsv ONLY, eval the FULL gen.tsv
    split (21000 examples). Runtime ~3s -- full-scale is fine here.
    Floor >= 0.98 (measured 99.75%, 20948/21000)."""
    sys.path.insert(0, str(REPO))
    from pure_vsa.cogs_hyperion import load_cogs_tsv
    from pure_vsa.cogs_template_learner import COGSTemplateLearner

    train = load_cogs_tsv(REPO / "data" / "cogs" / "raw" / "train.tsv")
    gen = load_cogs_tsv(REPO / "data" / "cogs" / "raw" / "gen.tsv")
    train_pairs = [(t[0], t[1]) for t in train]

    learner = COGSTemplateLearner()
    learner.fit(train_pairs)  # TRAIN split ONLY
    stats = learner.coverage_stats(gen)

    acc = stats["correct"] / len(gen)
    assert acc >= 0.98, (
        f"COGS gen accuracy regressed: {stats['correct']}/{len(gen)} = {acc:.4f} "
        "(floor 0.98, measured 99.75%)"
    )


def test_scan_hybrid_supervised_headline_floor(tmp_path):
    """SCAN addprim_jump supervised hybrid (per-token tagger + exact
    executor), full 7706-example held-out test split. Runtime ~20s train --
    full-scale is fine. Floor >= 0.95 held-out EM (measured 100%, 7706/7706).
    Runs the real producer script end-to-end (not a reimplementation) so it
    also pins the CLI/contract-JSON contract, not just the model."""
    out = tmp_path / "scan_addprim_jump__hybrid_tagger_supervised__seed0.json"
    subprocess.run(
        [PY, "-m", "experiments.scan_hybrid", "--n-eval", "7706", "--out", str(out)],
        cwd=str(REPO), check=True, timeout=120,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n"] == 7706
    assert result["accuracy"] >= 0.95, (
        f"SCAN supervised hybrid held-out EM regressed: {result['accuracy']:.4f} "
        "(floor 0.95, measured 100%)"
    )


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="scan_hybrid_outputonly REINFORCE full run takes >60s "
           "(measured ~150s under concurrent load); set RAINCG_RUN_SLOW_TESTS=1 to run",
)
def test_scan_hybrid_outputonly_headline_floor(tmp_path):
    """SCAN addprim_jump output-only REINFORCE hybrid, full 7706-example
    held-out test split. Skipped by default (slow, >60s); floor >= 0.95
    held-out EM (measured 100%, 7706/7706)."""
    out = tmp_path / "scan_addprim_jump__hybrid_outputonly_reinforce__seed0.json"
    subprocess.run(
        [PY, "-m", "experiments.scan_hybrid_outputonly", "--n-eval", "7706", "--out", str(out)],
        cwd=str(REPO), check=True, timeout=600,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n"] == 7706
    assert result["accuracy"] >= 0.95, (
        f"SCAN output-only REINFORCE hybrid held-out EM regressed: "
        f"{result['accuracy']:.4f} (floor 0.95, measured 100%)"
    )
