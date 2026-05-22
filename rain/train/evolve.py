# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/cognitive-kernel-polyglot/src/core/evolution.{rs,ts}
#                  + third_party/evolve/crates/evolve-core/src/* — Python port.
"""NSGA-II Pareto evolution + Bayesian beta-binomial champion/challenger.

Each variant has an N-dim fitness vector (e.g. [accuracy, calibration, FLOPs])
to be optimized along Pareto front. NSGA-II provides non-dominated sorting +
crowding-distance assignment for selection.

Promotion is gated by champion/challenger: a challenger only displaces the
current champion when its beta-binomial posterior over win-rate crosses a
configurable credible-interval threshold (default p_win >= 0.6).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np


# ---------- NSGA-II core ----------

def dominates(a: list[float], b: list[float]) -> bool:
    """Vector a Pareto-dominates b if a is no worse in any objective and strictly better in at least one.
    Convention: HIGHER is BETTER on every objective. (Caller negates FLOPs etc.)"""
    if len(a) != len(b):
        raise ValueError("fitness vectors must have same length")
    no_worse = all(ai >= bi for ai, bi in zip(a, b))
    strictly_better = any(ai > bi for ai, bi in zip(a, b))
    return no_worse and strictly_better


def non_dominated_sort(population: list[list[float]]) -> list[list[int]]:
    """Return a list of fronts. Each front is a list of indices into population."""
    n = len(population)
    dominates_count = [0] * n   # number of solutions dominating i
    dominated_by: list[list[int]] = [[] for _ in range(n)]   # solutions i dominates
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i]):
                dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)
    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)
    return fronts[:-1]  # drop trailing empty


def crowding_distance(population: list[list[float]], front: list[int]) -> dict[int, float]:
    """Crowding distance per individual within a single front."""
    if not front:
        return {}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    n_obj = len(population[front[0]])
    distance: dict[int, float] = {i: 0.0 for i in front}
    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda idx: population[idx][m])
        distance[sorted_front[0]] = math.inf
        distance[sorted_front[-1]] = math.inf
        f_min = population[sorted_front[0]][m]
        f_max = population[sorted_front[-1]][m]
        rng = f_max - f_min
        if rng == 0:
            continue
        for k in range(1, len(sorted_front) - 1):
            prev_v = population[sorted_front[k - 1]][m]
            next_v = population[sorted_front[k + 1]][m]
            distance[sorted_front[k]] += (next_v - prev_v) / rng
    return distance


# ---------- Genetic operators ----------

def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover: each gene from a or b 50/50."""
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b)


def mutate(genome: np.ndarray, rng: np.random.Generator, sigma: float = 0.1, rate: float = 0.05) -> np.ndarray:
    """Gaussian additive mutation. `rate` is the per-gene mutation probability."""
    mask = rng.random(genome.shape) < rate
    noise = rng.standard_normal(genome.shape) * sigma
    return genome + mask * noise


# ---------- Champion/challenger promotion ----------

@dataclass
class ChampionState:
    """Bayesian beta-binomial state for champion vs a challenger."""
    wins: int = 0       # times challenger beat champion in head-to-head trials
    losses: int = 0     # times champion beat challenger

    def posterior_win_rate(self, prior_a: float = 1.0, prior_b: float = 1.0) -> tuple[float, float]:
        """Returns (mean, stddev) of Beta(prior_a + wins, prior_b + losses)."""
        a = prior_a + self.wins
        b = prior_b + self.losses
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return mean, math.sqrt(var)

    def credible_lower_bound(self, prior_a: float = 1.0, prior_b: float = 1.0, z: float = 1.0) -> float:
        """Approximate one-sided credible lower bound at z stddevs (z=1.0 ~= 84%, z=1.96 ~= 97.5%)."""
        mean, sd = self.posterior_win_rate(prior_a, prior_b)
        return max(0.0, mean - z * sd)


def should_promote(state: ChampionState, threshold: float = 0.6, z: float = 1.0,
                   min_trials: int = 5) -> bool:
    """Promote challenger if the lower credible bound on its win rate exceeds `threshold`."""
    n_trials = state.wins + state.losses
    if n_trials < min_trials:
        return False
    return state.credible_lower_bound(z=z) >= threshold
