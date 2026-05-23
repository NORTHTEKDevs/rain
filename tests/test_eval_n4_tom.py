# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from evals.tier1_novelty.tom import run_benchmark


def test_n4_sally_anne_10_of_10():
    r = run_benchmark()
    assert r.sally_anne_correct == r.sally_anne_total == 10
    assert r.sally_anne_pass


def test_n4_bdi_at_least_18_of_20():
    r = run_benchmark()
    assert r.bdi_correct >= 18, f"BDI: {r.bdi_correct}/{r.bdi_total}, failures: {r.failed_cases}"
    assert r.bdi_pass


def test_n4_overall_pass():
    r = run_benchmark()
    assert r.overall_pass
