# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
import pytest
from rain.cognition.metacog import CalibrationTally, classify_epistemic, Thresholds


def test_calibration_starts_at_prior_mean():
    tally = CalibrationTally(prior_correct=1.0, prior_total=2.0)
    # uniform prior gives 0.5
    assert tally.calibration("capital_of") == 0.5


def test_update_moves_calibration_toward_outcome():
    tally = CalibrationTally(prior_correct=1.0, prior_total=2.0)
    for _ in range(100):
        tally.update("capital_of", was_correct=True)
    # With 100 correct + uniform prior we should be near 1.0
    assert tally.calibration("capital_of") > 0.95


def test_classify_unknown_below_guess_threshold():
    assert classify_epistemic(0.1) == "unknown"


def test_classify_know_requires_high_calibration():
    # High confidence but no relation calibration -> "think", not "know"
    assert classify_epistemic(0.9, relation_calibration=None) == "think"
    # High confidence AND high calibration -> "know"
    assert classify_epistemic(0.9, relation_calibration=0.95) == "know"


def test_classify_levels():
    assert classify_epistemic(0.95, relation_calibration=0.95) == "know"
    assert classify_epistemic(0.7, relation_calibration=0.95) == "think"
    assert classify_epistemic(0.3, relation_calibration=0.95) == "guess"
    assert classify_epistemic(0.1, relation_calibration=0.95) == "unknown"
