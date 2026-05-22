# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from evals.tier1_novelty.retention import run_benchmark


def test_n1_retention_above_threshold():
    """Headline Tier-1 surface: A facts must survive a full sweep of B-interference.

    Scaled down for test speed; full sweep (5K + 5K) runs from `python -m
    evals.tier1_novelty.retention`.
    """
    r = run_benchmark(n_a=200, n_b=200, n_probe=80, dim=2048, num_shards=16, seed=0)
    assert r.initial_recall > 0.0, (
        f"initial_recall is 0 -- the model could not retrieve its own seeded facts "
        f"(result: {r})"
    )
    assert r.retention >= r.pass_threshold, (
        f"retention {r.retention:.3f} < {r.pass_threshold} "
        f"(initial={r.initial_recall:.3f}, post_b={r.post_b_recall:.3f})"
    )
    assert r.overall_pass
    assert not r.kill_triggered


def test_n1_no_decay_when_no_interference():
    """Sanity: if we don't add B facts, retention should be 1.0 (model still
    retrieves A perfectly when re-asked).
    """
    r = run_benchmark(n_a=200, n_b=0, n_probe=80, dim=2048, num_shards=16, seed=0)
    assert r.retention == 1.0, (
        f"retention should be 1.0 with no interference, got {r.retention}"
    )


def test_n1_record_per_query():
    """When --record-per-query is set, the result carries per-query rows for
    both phases.
    """
    r = run_benchmark(
        n_a=50, n_b=50, n_probe=20, dim=1024, num_shards=8, seed=0,
        record_per_query=True,
    )
    assert len(r.per_query) == 40  # 20 initial + 20 post_b
    phases = {row["phase"] for row in r.per_query}
    assert phases == {"initial", "post_b"}
