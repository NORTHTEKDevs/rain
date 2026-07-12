from evals.tier1_novelty.calibration import run_benchmark


def test_n2_ece_below_threshold_default():
    """With all facts seeded, ECE should be very low (the model is direct on every Q)."""
    r = run_benchmark(n=200, miss_seed_rate=0.0, seed=0)  # 200 for test speed
    assert r.ece <= 0.05, f"ECE {r.ece}, bins: {r.bin_stats}"
    assert r.overall_pass


def test_n2_handles_miss_seed_rate():
    """With 20% facts missing, model should refuse correctly and ECE stay low
    because refusals are confidence=0.0 (and they're 'correct' refusals when fact omitted)."""
    r = run_benchmark(n=200, miss_seed_rate=0.2, seed=0)
    # ECE should still be low because the model correctly refuses + correctly answers
    assert r.ece <= 0.10, f"ECE {r.ece}, bins: {r.bin_stats}"
