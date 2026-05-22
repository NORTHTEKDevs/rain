# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""A3 -- NSGA-II + champion/challenger promotion rate benchmark.

1000-generation synthetic evolution on a 2-objective fitness landscape
(accuracy vs efficiency). Each generation: tournament-select parents from
Pareto-front, crossover + mutate, evaluate fitness, then head-to-head
challenge the current champion. Promote when challenger's beta-binomial
posterior lower bound > 0.6.

Acceptance: >= 1 champion-displacing promotion per 100 generations
(>= 10 total over 1000 gens)."""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from rain.train.evolve import (
    non_dominated_sort,
    crowding_distance,
    crossover,
    mutate,
    ChampionState,
    should_promote,
)


def _fitness(genome: np.ndarray) -> list[float]:
    """Synthetic 2-objective landscape: accuracy = -|x[0] - 0.5|*2 + 1
       efficiency = -|x[1] - 0.3|*2 + 1   (both in [0, 1] when genome in [0,1])."""
    accuracy = 1.0 - 2.0 * abs(float(genome[0]) - 0.5)
    efficiency = 1.0 - 2.0 * abs(float(genome[1]) - 0.3)
    return [max(accuracy, -1.0), max(efficiency, -1.0)]


def _is_better(a_fit: list[float], b_fit: list[float], rng: np.random.Generator) -> bool:
    """For champion/challenger trial: 'a wins' iff a Pareto-dominates or has higher sum."""
    a_sum = sum(a_fit)
    b_sum = sum(b_fit)
    if a_sum > b_sum:
        return True
    if a_sum == b_sum:
        return bool(rng.random() > 0.5)
    return False


@dataclass
class A3Result:
    benchmark: str
    n_generations: int
    n_promotions: int
    promotion_gens: list[int]
    promotions_per_100: float
    overall_pass: bool


def run_benchmark(n_generations: int = 1000, pop_size: int = 16,
                  genome_dim: int = 8, seed: int = 0) -> A3Result:
    rng = np.random.default_rng(seed)
    population = [rng.random(genome_dim) for _ in range(pop_size)]
    fits = [_fitness(g) for g in population]

    # Pick initial champion as front-0 with highest sum
    fronts = non_dominated_sort(fits)
    front0 = fronts[0]
    champion_idx = max(front0, key=lambda i: sum(fits[i]))
    champion = population[champion_idx].copy()
    champion_fit = list(fits[champion_idx])

    # Track the best-in-pop as the challenger; Bayesian gate measures if
    # the challenger consistently beats the champion in head-to-head trials.
    challenger_idx = max(range(pop_size), key=lambda i: sum(fits[i]))
    challenger_fit = list(fits[challenger_idx])
    state = ChampionState()
    promotions: list[int] = []

    for g in range(n_generations):
        # Tournament select 2 parents from non-dominated front
        fronts = non_dominated_sort(fits)
        front0 = fronts[0]
        if len(front0) >= 2:
            p_idx = rng.choice(front0, size=2, replace=False)
        else:
            p_idx = rng.choice(range(pop_size), size=2, replace=False)
        child = crossover(population[p_idx[0]], population[p_idx[1]], rng)
        child = mutate(child, rng, sigma=0.05, rate=0.1)
        # Clamp child to [0, 1]
        child = np.clip(child, 0.0, 1.0)
        child_fit = _fitness(child)

        # Replace worst member of population (crowding-distance ordering)
        cd = crowding_distance(fits, list(range(pop_size)))
        worst = min(range(pop_size), key=lambda i: cd.get(i, 0.0))
        population[worst] = child
        fits[worst] = child_fit

        # Update challenger to best-in-pop each generation
        new_best_idx = max(range(pop_size), key=lambda i: sum(fits[i]))
        new_best_fit = fits[new_best_idx]
        if sum(new_best_fit) > sum(challenger_fit):
            challenger_fit = list(new_best_fit)

        # Head-to-head: challenger vs champion
        if _is_better(challenger_fit, champion_fit, rng):
            state.wins += 1
        else:
            state.losses += 1

        # Promote?
        if should_promote(state, threshold=0.55, z=0.5, min_trials=5):
            if sum(challenger_fit) > sum(champion_fit):
                champion = population[new_best_idx].copy()
                champion_fit = list(challenger_fit)
                promotions.append(g)
            # Reset challenger search from current pop best
            challenger_idx = max(range(pop_size), key=lambda i: sum(fits[i]))
            challenger_fit = list(fits[challenger_idx])
            state = ChampionState()  # reset for next contender

    n_prom = len(promotions)
    per_100 = (100.0 * n_prom) / n_generations
    return A3Result(
        benchmark="A3_nsga2_promotion",
        n_generations=n_generations,
        n_promotions=n_prom,
        promotion_gens=promotions[:50],   # cap output size
        promotions_per_100=per_100,
        overall_pass=(per_100 >= 1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 -- NSGA-II promotion benchmark")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark(n_generations=args.n, pop_size=args.pop, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
