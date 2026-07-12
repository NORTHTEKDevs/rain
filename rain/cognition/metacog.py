"""Per-relation Bayesian calibration tally + epistemic classification.

Tracks for each relation type a running count of (correct, total) outcomes,
plus a Bayesian beta-binomial posterior over the per-relation true-accuracy
parameter. Calibration weight = posterior mean (shrunk toward prior).

Epistemic class assigned to each output:
  - know:    confidence > tau_know and relation-calibration > kappa_know
  - think:   tau_think < confidence <= tau_know
  - guess:   tau_guess < confidence <= tau_think
  - unknown: confidence <= tau_guess or no candidates available
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    tau_know: float = 0.85
    tau_think: float = 0.50
    tau_guess: float = 0.20
    kappa_know: float = 0.85


class CalibrationTally:
    def __init__(self, prior_correct: float = 1.0, prior_total: float = 2.0) -> None:
        """Beta(prior_correct, prior_total - prior_correct) prior. Defaults Beta(1,1) = uniform."""
        self.prior_correct = prior_correct
        self.prior_total = prior_total
        self._correct: dict[str, int] = defaultdict(int)
        self._total: dict[str, int] = defaultdict(int)

    def update(self, relation: str, was_correct: bool) -> None:
        self._total[relation] += 1
        if was_correct:
            self._correct[relation] += 1

    def calibration(self, relation: str) -> float:
        """Posterior mean = (prior_correct + correct) / (prior_total + total)."""
        c = self._correct.get(relation, 0)
        t = self._total.get(relation, 0)
        return (self.prior_correct + c) / (self.prior_total + t)


def classify_epistemic(
    confidence: float,
    relation_calibration: float | None = None,
    thresholds: Thresholds | None = None,
) -> str:
    """Classify an output as know / think / guess / unknown."""
    t = thresholds or Thresholds()
    if confidence <= t.tau_guess:
        return "unknown"
    if relation_calibration is None or relation_calibration <= t.kappa_know:
        # No relation info -- classify by confidence alone, but never claim "know" without high relation calibration
        if confidence > t.tau_know:
            return "think"  # high confidence but no relation backing -> "think" not "know"
        if confidence > t.tau_think:
            return "think"
        return "guess"
    # have high relation calibration
    if confidence > t.tau_know:
        return "know"
    if confidence > t.tau_think:
        return "think"
    return "guess"
