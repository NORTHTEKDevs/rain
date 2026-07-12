from raincg.bench.common import exact_match, ExactMatchResult, wilson_ci


def test_exact_match_counts():
    preds = [["A", "B"], ["C"], ["X"]]
    golds = [["A", "B"], ["C"], ["Y"]]
    r = exact_match(preds, golds)
    assert isinstance(r, ExactMatchResult)
    assert r.correct == 2
    assert r.total == 3
    assert abs(r.accuracy - 2 / 3) < 1e-9


def test_exact_match_length_mismatch_is_wrong():
    r = exact_match([["A"]], [["A", "B"]])
    assert r.correct == 0


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(8, 10)
    assert abs(lo - 0.49) < 0.01
    assert abs(hi - 0.94) < 0.01


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_full_and_empty():
    lo, hi = wilson_ci(0, 5)
    assert lo == 0.0
    assert 0.0 < hi < 1.0
    lo2, hi2 = wilson_ci(5, 5)
    assert hi2 == 1.0
    assert 0.0 < lo2 < 1.0


def test_wilson_ci_bounds_are_ordered_and_clamped():
    lo, hi = wilson_ci(3, 7)
    assert 0.0 <= lo <= hi <= 1.0
