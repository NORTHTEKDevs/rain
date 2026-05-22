# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from evals.tier1_novelty.transparency import run_benchmark


def test_n5_citation_coverage_98pct():
    r = run_benchmark()
    assert r.citation_coverage >= 0.98, f"coverage {r.citation_coverage}, failures: {r.failures}"
    assert r.citation_pass


def test_n5_hallucination_rate_1pct():
    r = run_benchmark()
    assert r.hallucination_rate <= 0.01, f"halluc rate {r.hallucination_rate}, failures: {r.failures}"
    assert r.hallucination_pass


def test_n5_overall_pass():
    r = run_benchmark()
    assert r.overall_pass
