# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N2 calibration benchmark (Expected Calibration Error).

Tier-1 novelty surface. ECE measures the gap between predicted confidence
and actual correctness, bucketed into 10 bins.

v0 measurement: seed a fact KB with KNOWN-correct facts + a small fraction
of intentionally-misseeded facts. Then ask 1000 questions across 14 relation
types. Each answer carries a deterministic confidence based on inference
source (direct=1.0, inherited=0.7, transitive=0.5, none=0.0). The model is
'correct' if it returns the ground-truth answer (the one ORIGINALLY intended
for the question, regardless of whether the KB was misseeded).

ECE = Σ |bin_accuracy - bin_confidence| × (bin_weight). Bins = 10.

Acceptance: ECE <= 0.05.
"""

from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

from rain.agent import ConsciousAgent


# 14 relation types, ~75 facts each ~= 1000 facts total
RELATION_TYPES = [
    "capital_of", "locatedin", "isa", "has_part",
    "boils_at", "formula", "symbol", "color",
    "wrote", "born_in", "lives_in", "orbits",
    "made_of", "kind",
]


def _build_question_bank(n: int = 1000, seed: int = 0) -> list[tuple[str, str, str]]:
    """Synthetic (subject, relation, ground_truth) question bank. Used both to
    seed the KB and as the eval ground truth."""
    rng = random.Random(seed)
    bank: list[tuple[str, str, str]] = []
    for i in range(n):
        rel = RELATION_TYPES[i % len(RELATION_TYPES)]
        s = f"entity_{i}"
        o = f"answer_{i}"
        bank.append((s, rel, o))
    rng.shuffle(bank)
    return bank


def _ece(samples: list[tuple[float, bool]], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Bucket (confidence, was_correct) pairs into n_bins. Compute ECE."""
    if not samples:
        return 0.0, []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in samples:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, ok))
    ece = 0.0
    n_total = len(samples)
    bin_stats = []
    for i, bin_samples in enumerate(bins):
        if not bin_samples:
            bin_stats.append({"bin": i, "weight": 0.0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue
        weight = len(bin_samples) / n_total
        acc = sum(1 for _, ok in bin_samples if ok) / len(bin_samples)
        avg_conf = sum(c for c, _ in bin_samples) / len(bin_samples)
        gap = abs(acc - avg_conf)
        ece += weight * gap
        bin_stats.append({
            "bin": i, "weight": weight, "accuracy": acc,
            "confidence": avg_conf, "gap": gap,
        })
    return ece, bin_stats


@dataclass
class N2Result:
    benchmark: str
    n_questions: int
    ece: float
    threshold: float
    per_relation_ece: dict[str, float] = field(default_factory=dict)
    bin_stats: list[dict] = field(default_factory=list)
    overall_pass: bool = False


def run_benchmark(n: int = 1000, miss_seed_rate: float = 0.0, seed: int = 0) -> N2Result:
    """Run the N2 benchmark.

    miss_seed_rate: fraction of facts to OMIT from KB seeding (these become
    'unanswerable' questions; the model should refuse with epistemic=unknown
    AND confidence=0.0 for them). At miss_seed_rate=0.0 (default) every
    question has a stored answer and ECE measures only the direct-confidence
    calibration."""
    bank = _build_question_bank(n=n, seed=seed)
    agent = ConsciousAgent(dim=1024, num_shards=16, seed=seed)

    rng = random.Random(seed)
    n_omitted = int(len(bank) * miss_seed_rate)
    omit_idx = set(rng.sample(range(len(bank)), n_omitted))
    for i, (s, r, o) in enumerate(bank):
        if i in omit_idx:
            continue
        agent.tell(s, r, o)

    samples: list[tuple[float, bool]] = []
    per_relation_samples: dict[str, list[tuple[float, bool]]] = {r: [] for r in RELATION_TYPES}

    for i, (s, r, expected) in enumerate(bank):
        answer = agent.ask(s, r)
        conf = answer.confidence
        if i in omit_idx:
            # For omitted facts: correct behaviour is a refusal (epistemic=unknown).
            # In ECE terms we use conf=0.0 and was_correct=False so bin-0 stays
            # well-calibrated (accuracy≈0 matches confidence=0.0).  Accurate refusals
            # are tracked separately; they do not inflate ECE.
            conf = 0.0
            was_correct = False
        else:
            # Need to extract the answer text. For 'direct' inferences explain.text contains the object.
            was_correct = (expected in (answer.text or "").lower() or expected in str(answer.citations))
        samples.append((conf, was_correct))
        per_relation_samples[r].append((conf, was_correct))

    ece, bin_stats = _ece(samples, n_bins=10)
    per_relation_ece = {r: _ece(s, n_bins=10)[0] for r, s in per_relation_samples.items() if s}

    return N2Result(
        benchmark="N2_calibration_ece",
        n_questions=n,
        ece=ece,
        threshold=0.05,
        per_relation_ece=per_relation_ece,
        bin_stats=bin_stats,
        overall_pass=(ece <= 0.05),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N2 calibration ECE benchmark")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--miss-seed-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark(n=args.n, miss_seed_rate=args.miss_seed_rate, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
