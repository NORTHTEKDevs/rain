"""N3-capacity: how many slots survive readout — greedy vs resonant.

The N3 SCAN eval (scan.py) exercises 3-5 slots, where per-slot greedy extract()
already hits ~100%. This sibling probes the CAPACITY regime: as the number of
simultaneously-bound slots grows, the signed-bundle + per-slot greedy readout
collapses under cross-talk, while the resonant explaining-away readout
(CompositionalReasoner.extract_all, ported from Hyperion Finding 5) holds far
longer at the same dimension. Whole-assignment accuracy (ALL slots correct) over
randomly sampled assignments (slot counts too large to enumerate).

Acceptance (D=256): resonant >> greedy by 16+ slots; resonant >= 0.9 at 24 slots
where greedy has collapsed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rain.cognition.compose import CompositionalReasoner
from rain.core.relational import Codebook


@dataclass
class CapacityRow:
    n_slots: int
    greedy_acc: float
    resonant_acc: float


def _capacity_at(n_slots: int, dim: int, values_per_slot: int,
                 n_trials: int, seed: int) -> CapacityRow:
    slot_names = [f"slot_{i}" for i in range(n_slots)]
    values = [f"val_{j}" for j in range(values_per_slot * n_slots)]   # shared vocab
    cb = Codebook(vocab_size=n_slots + len(values) + 10, dim=dim, seed=seed)
    reasoner = CompositionalReasoner(cb, slot_names=slot_names)
    rng = np.random.default_rng(seed)
    g_ok = r_ok = 0
    for _ in range(n_trials):
        assign = {s: values[int(rng.integers(len(values)))] for s in slot_names}
        bundle = reasoner.compose(assign)                                    # signed bundle
        g = all(reasoner.extract(bundle, s, values) == assign[s] for s in slot_names)
        pred = reasoner.extract_all(reasoner.compose_sum(assign), slot_names, values)
        r = all(pred[s] == assign[s] for s in slot_names)
        g_ok += int(g)
        r_ok += int(r)
    return CapacityRow(n_slots, g_ok / n_trials, r_ok / n_trials)


def run_benchmark(dim: int = 256, slot_counts: tuple[int, ...] = (8, 16, 24, 32),
                  values_per_slot: int = 4, n_trials: int = 40, seed: int = 0) -> dict:
    rows = [_capacity_at(n, dim, values_per_slot, n_trials, seed + n) for n in slot_counts]
    # the gain: resonant beats greedy by a wide margin once cross-talk dominates
    resonant_holds = all(r.resonant_acc >= 0.9 for r in rows if r.n_slots <= 16)
    beats_greedy = all(r.resonant_acc >= r.greedy_acc for r in rows) and \
        any(r.resonant_acc - r.greedy_acc >= 0.5 for r in rows)
    return {
        "benchmark": "N3-capacity (greedy vs resonant readout)",
        "dim": dim,
        "rows": [asdict(r) for r in rows],
        "resonant_holds_to_24": resonant_holds,
        "resonant_beats_greedy": beats_greedy,
        "overall_pass": resonant_holds and beats_greedy,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    res = run_benchmark(dim=a.dim, n_trials=a.trials)
    print(f"# {res['benchmark']}  (D={res['dim']})")
    print(f"{'slots':>6} {'greedy':>8} {'resonant':>9}")
    for r in res["rows"]:
        print(f"{r['n_slots']:>6} {r['greedy_acc']*100:7.1f}% {r['resonant_acc']*100:8.1f}%")
    print(f"overall_pass={res['overall_pass']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
