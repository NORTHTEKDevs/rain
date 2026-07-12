"""Shared benchmark utilities: exact-match scoring."""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class ExactMatchResult:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def exact_match(preds: list[list[str]], golds: list[list[str]]) -> ExactMatchResult:
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return ExactMatchResult(correct=correct, total=len(golds))


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. n<=0 -> (0.0, 1.0)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    adj = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))
