# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Test for the N3-capacity eval: resonant readout vastly outscales greedy."""

from __future__ import annotations

from evals.tier1_novelty.scan_capacity import run_benchmark


def test_resonant_readout_outscales_greedy():
    res = run_benchmark(dim=256, slot_counts=(8, 16, 24), n_trials=30, seed=0)
    rows = {r["n_slots"]: r for r in res["rows"]}
    # at 16 slots the signed-bundle greedy readout has collapsed; resonant holds
    assert rows[16]["greedy_acc"] <= 0.2
    assert rows[16]["resonant_acc"] >= 0.9
    assert res["resonant_beats_greedy"]
    assert res["overall_pass"]
