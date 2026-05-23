# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import math

import numpy as np

from rain.train.evolve import (
    ChampionState,
    crossover,
    crowding_distance,
    dominates,
    mutate,
    non_dominated_sort,
    should_promote,
)


def test_dominates_basic():
    assert dominates([1.0, 1.0], [0.5, 0.5])
    assert not dominates([1.0, 0.0], [0.5, 1.0])  # neither dominates


def test_non_dominated_sort_three_fronts():
    pop = [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
    fronts = non_dominated_sort(pop)
    assert fronts[0] == [0]
    assert fronts[1] == [1]
    assert fronts[2] == [2]


def test_crowding_distance_boundary_is_inf():
    pop = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    cd = crowding_distance(pop, [0, 1, 2])
    # Boundary points (extremes) are infinite
    assert cd[0] == math.inf
    assert cd[2] == math.inf
    assert cd[1] < math.inf


def test_crossover_mutate_produce_in_range_values():
    rng = np.random.default_rng(0)
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([0.6, 0.7, 0.8, 0.9])
    c = crossover(a, b, rng)
    for v in c:
        assert 0.1 <= v <= 0.9
    m = mutate(c, rng, sigma=0.01, rate=1.0)
    assert m.shape == c.shape


def test_should_promote_below_threshold():
    s = ChampionState(wins=2, losses=8)
    assert not should_promote(s, threshold=0.6, z=1.0, min_trials=5)


def test_should_promote_above_threshold():
    s = ChampionState(wins=18, losses=2)
    assert should_promote(s, threshold=0.6, z=1.0, min_trials=5)
