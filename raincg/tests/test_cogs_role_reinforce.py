"""Smoke test for experiments/cogs_role_reinforce.py -- fast, tiny-step,
deterministic (seed=0) check that: the output JSON matches the SHARED
CONTRACT schema; the honest (output-only REINFORCE) run recovers high
accuracy on the ablation-sensitive target category
(raincg/cogs_intrans_role_ablation_20260711.json); and both negative
controls (shuffled-target, label-flip) fall well short of it and stay below
the design's 44.4%-floor acceptance bar (raincg/COGS-ATTACK-DESIGN.md,
kill criteria section 3).

Note on thresholds: the shuffled-target control does NOT converge to
literal near-zero accuracy on the target category -- diagnosed (see the
script's inline comments and the 2026-07-11 spike report) as a majority-
class collapse (it learns to predict the training pool's dominant role,
~75% agent by token count) rather than genuine per-verb correspondence.
Downstream category accuracy for that collapsed table lands ~28-31% at
this step count, well below the honest run and below the 44.4% floor, but
not "near zero" in the literal sense -- the assertions below reflect the
actually-measured, stable behavior rather than an unverified expectation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

REQUIRED_KEYS = {
    "bench", "system", "split", "n", "correct", "accuracy", "ci95",
    "params", "train_s", "eval_s", "seed", "config", "timestamp",
    "evidence_tier", "notes",
}

TARGET_CAT = "only_seen_as_transitive_subj_as_unacc_subj"


def test_cogs_role_reinforce_tiny_run(tmp_path):
    out = tmp_path / "cogs_gen__template_learner_reinforce_role__seed0.json"
    r = subprocess.run(
        [PY, "-m", "experiments.cogs_role_reinforce",
         "--steps", "50", "--batch", "64", "--seed", "0", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    d = json.loads(out.read_text())

    assert REQUIRED_KEYS <= set(d)
    assert d["bench"] == "cogs_gen"
    assert d["system"] == "template_learner_reinforce_role"
    assert d["split"] == "gen"
    assert d["seed"] == 0
    assert isinstance(d["n"], int) and d["n"] == 21000
    assert 0 <= d["correct"] <= d["n"]
    assert 0.0 <= d["accuracy"] <= 1.0
    assert len(d["ci95"]) == 2
    assert d["params"] > 0  # 61 verb types x 2 roles

    cfg = d["config"]
    assert "negative_controls" in cfg
    assert "shuffled_target" in cfg["negative_controls"]
    assert "label_flip" in cfg["negative_controls"]
    assert TARGET_CAT in cfg["per_category_ablation_sensitive"]

    honest_acc = cfg["per_category_ablation_sensitive"][TARGET_CAT]["acc"]
    shuffled_acc = cfg["negative_controls"]["shuffled_target"][TARGET_CAT]["acc"]
    flipped_acc = cfg["negative_controls"]["label_flip"][TARGET_CAT]["acc"]

    # Honest run recovers the ablation-sensitive category well above the
    # measured 4.4% ablated floor.
    assert honest_acc > 0.9

    # Both negative controls must fail to reach the honest run's accuracy on
    # the same category and stay below the design's 44.4% acceptance floor
    # -- a control that clears this would void the positive result
    # (raincg/COGS-ATTACK-DESIGN.md kill criteria).
    assert shuffled_acc < 0.45
    assert flipped_acc < 0.45
    assert shuffled_acc < honest_acc - 0.3
    assert flipped_acc < honest_acc - 0.3

    # Table-level diagnostic (lower-variance than downstream category
    # accuracy): the honest run should reproduce the gold-label-scan
    # oracle's per-verb table closely; label-flip should anti-correlate.
    oracle_ref = cfg["gold_label_scan_oracle_reference"]
    assert oracle_ref["table_agreement_with_reinforce_table"] > 0.9
    assert cfg["negative_controls"]["label_flip"]["table_agreement_with_oracle"] < 0.1
