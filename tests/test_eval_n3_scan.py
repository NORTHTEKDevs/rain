# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N3 SCAN benchmark sanity test."""

from evals.tier1_novelty.scan import run_benchmark


def test_n3_scan_passes_at_default_dim():
    """At D=2048 the benchmark should pass cleanly."""
    result = run_benchmark(dim=2048, seed=0)
    assert result.add_primitive_acc >= 1.0, f"add-primitive {result.add_primitive_acc}"
    assert result.slot3_acc >= 0.95, f"3-slot {result.slot3_acc}"
    assert result.slot4_acc >= 0.90, f"4-slot {result.slot4_acc}"
    assert result.slot5_acc >= 0.85, f"5-slot {result.slot5_acc}"
    assert result.overall_pass


def test_n3_low_dim_fails_gracefully():
    """At D=128 the cleanup HV is too noisy -- accuracy should drop, but no crash."""
    result = run_benchmark(dim=128, seed=0)
    # We DON'T assert pass here -- just that the result is well-formed
    assert 0.0 <= result.add_primitive_acc <= 1.0
    assert 0.0 <= result.slot5_acc <= 1.0
