from evals.tier3_soundness.nsga2_promotion import run_benchmark


def test_a3_promotion_rate_passes():
    """Over 200 gens we expect >= 2 promotions (i.e. >=1 per 100)."""
    r = run_benchmark(n_generations=200, pop_size=12, seed=0)
    assert r.promotions_per_100 >= 1.0, f"per_100={r.promotions_per_100}, promos={r.n_promotions}"
    assert r.overall_pass


def test_a3_result_well_formed():
    r = run_benchmark(n_generations=50, pop_size=8, seed=0)
    assert r.benchmark == "A3_nsga2_promotion"
    assert r.n_generations == 50
    assert isinstance(r.promotion_gens, list)
