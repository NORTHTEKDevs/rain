# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from evals.tier3_soundness.efe_source_mix import run_benchmark


def test_a1_each_non_hymn_source_contributes():
    r = run_benchmark(n_prompts=50, seed=0)  # 50 for speed
    # Each non-HYMN source contributes >=5% of total weighted fusion
    for src in ("kb", "lsm", "bigram", "fep", "tsetlin", "crystal"):
        assert (
            r.source_contribution_pct.get(src, 0.0) >= 5.0
        ), f"{src}: {r.source_contribution_pct.get(src, 0.0)}% < 5%"
    assert r.overall_pass


def test_a1_pct_sums_approx_100():
    r = run_benchmark(n_prompts=20, seed=1)
    total = sum(r.source_contribution_pct.values())
    assert abs(total - 100.0) < 0.01
