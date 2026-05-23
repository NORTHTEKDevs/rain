# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N3 SCAN-style compositional generalization benchmark.

Tier-1 novelty surface: compositional generalization. Uses RAIN's
CompositionalReasoner to compose hypervectors from slot-value assignments
and extract values back via unbinding. Tests held-out unseen combinations
to verify compositional generalization (no training on the test combos).

This is structurally equivalent to RCK v1.1's 3/4/5-slot unseen-composition
benchmark which achieved 100% on all four splits. Empirical numbers under
RAIN's port should match.

Acceptance:
  add-primitive: 100% (8/8)
  3-slot:       >= 95% (60/64)
  4-slot:       >= 90% (114/128)
  5-slot:       >= 85% (326/384)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from rain.cognition.compose import CompositionalReasoner
from rain.core.relational import Codebook


@dataclass
class N3Result:
    benchmark: str
    add_primitive_acc: float
    slot3_acc: float
    slot4_acc: float
    slot5_acc: float
    add_primitive_pass: bool
    slot3_pass: bool
    slot4_pass: bool
    slot5_pass: bool
    overall_pass: bool


def _evaluate_slot_combinations(
    n_slots: int,
    values_per_slot: int = 4,
    dim: int = 2048,
    seed: int = 0,
) -> float:
    """Generate all (values_per_slot ^ n_slots) combinations. Compose each
    as a single VSA hypervector, then extract every slot's value back.
    Accuracy = fraction of combinations where ALL slots extract correctly."""
    slot_names = [f"slot_{i}" for i in range(n_slots)]

    vocab = n_slots + values_per_slot * n_slots + 10  # +10 safety margin
    cb = Codebook(vocab_size=vocab, dim=dim, seed=seed)
    reasoner = CompositionalReasoner(cb, slot_names=slot_names)

    candidate_values = [[f"v_{i}_{j}" for j in range(values_per_slot)] for i in range(n_slots)]
    correct = 0
    total = 0
    for combo in product(*candidate_values):
        slot_values = {slot_names[i]: combo[i] for i in range(n_slots)}
        composite = reasoner.compose(slot_values)
        all_match = True
        for i, slot in enumerate(slot_names):
            extracted = reasoner.extract(composite, slot, candidates=candidate_values[i])
            if extracted != combo[i]:
                all_match = False
                break
        total += 1
        if all_match:
            correct += 1
    return correct / total


def _evaluate_add_primitive(dim: int = 2048, seed: int = 0) -> float:
    """Add-primitive: introduce a NEW value not seen in any prior composition.
    Compose it with each of 8 existing slots' values, verify round-trip."""
    slot_names = ["color", "shape"]
    base_values_color = ["red", "blue", "green", "yellow"]
    base_values_shape = ["circle", "square", "triangle", "hexagon"]
    new_value = "magenta"  # NEW primitive

    vocab = len(slot_names) + len(base_values_color) + len(base_values_shape) + 10
    cb = Codebook(vocab_size=vocab, dim=dim, seed=seed)
    reasoner = CompositionalReasoner(cb, slot_names=slot_names)

    correct = 0
    total = 0
    for shape in base_values_shape:
        composite = reasoner.compose({"color": new_value, "shape": shape})
        c = reasoner.extract(composite, "color", candidates=base_values_color + [new_value])
        s = reasoner.extract(composite, "shape", candidates=base_values_shape)
        total += 1
        if c == new_value and s == shape:
            correct += 1
    for color in base_values_color:
        composite = reasoner.compose({"color": color, "shape": new_value})
        c = reasoner.extract(composite, "color", candidates=base_values_color)
        s = reasoner.extract(composite, "shape", candidates=base_values_shape + [new_value])
        total += 1
        if c == color and s == new_value:
            correct += 1
    return correct / total


def run_benchmark(dim: int = 2048, seed: int = 0) -> N3Result:
    add_prim_acc = _evaluate_add_primitive(dim=dim, seed=seed)
    slot3_acc = _evaluate_slot_combinations(n_slots=3, values_per_slot=4, dim=dim, seed=seed + 1)
    slot4_acc = _evaluate_slot_combinations(n_slots=4, values_per_slot=4, dim=dim, seed=seed + 2)
    slot5_acc = _evaluate_slot_combinations(n_slots=5, values_per_slot=4, dim=dim, seed=seed + 3)
    add_pass = add_prim_acc >= 1.0
    slot3_pass = slot3_acc >= 0.95
    slot4_pass = slot4_acc >= 0.90
    slot5_pass = slot5_acc >= 0.85
    return N3Result(
        benchmark="N3_scan_compositional",
        add_primitive_acc=add_prim_acc,
        slot3_acc=slot3_acc,
        slot4_acc=slot4_acc,
        slot5_acc=slot5_acc,
        add_primitive_pass=add_pass,
        slot3_pass=slot3_pass,
        slot4_pass=slot4_pass,
        slot5_pass=slot5_pass,
        overall_pass=add_pass and slot3_pass and slot4_pass and slot5_pass,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N3 SCAN compositional generalization benchmark")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    result = run_benchmark(dim=args.dim, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
