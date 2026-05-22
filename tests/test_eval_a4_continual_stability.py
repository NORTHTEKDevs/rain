# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import pytest
from evals.tier3_soundness.continual_stability import run_benchmark


def test_a4_short_run_no_nan_no_corruption():
    """1000-turn smoke run -- much faster than 10K but still proves stability."""
    r = run_benchmark(n_turns=1000, dim=256, seed=0)
    assert r.no_nan, f"NaN issues: {r.issues}"
    assert r.first_50_retention >= 0.95, f"retention {r.first_50_retention}"
    assert r.ece_drift <= 0.10, f"ECE drift {r.ece_drift}"
    assert r.overall_pass


@pytest.mark.slow
def test_a4_full_10k_turns():
    """Full 10K-turn run -- slow, marked as 'slow' for normal CI skip."""
    r = run_benchmark(n_turns=10000, dim=512, seed=0)
    assert r.overall_pass, (
        f"issues: {r.issues}, retention: {r.first_50_retention}, ece_drift: {r.ece_drift}"
    )
