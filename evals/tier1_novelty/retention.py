# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N1 continual-learning retention benchmark (A -> B -> A).

Tier-1 novelty surface. Measures whether the model retains task-A facts
after continual training on task-B facts. LLMs forget under this protocol
(CSUR 2025: frontier models < 0.30). RAIN's sharded HRR KB stores each
fact as an explicit bind+bundle into a hypervector shard, with no global
weight overwrite, so retention should hold structurally.

Procedure (per docs/architecture/benchmark-suite.md):
1. Two disjoint task corpora.
   A = animals + habitats (subject = animal, relation = lives_in, object = habitat).
   B = chemistry + composition (subject = compound, relation = made_of, object = element).
2. tell() all A facts.
3. ask() a sample of A facts -> initial_recall.
4. tell() all B facts (interference).
5. ask() the SAME sample of A facts -> post_b_recall.
6. retention = post_b_recall / initial_recall (1.0 if initial_recall is 0).

Acceptance: retention >= 0.50. Kill criterion: retention < 0.40.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rain.agent import ConsciousAgent

_ANIMALS = [
    "lion", "tiger", "wolf", "bear", "fox", "eagle", "shark", "dolphin",
    "elephant", "giraffe", "zebra", "panda", "koala", "kangaroo", "owl",
    "hawk", "penguin", "whale", "seal", "otter", "rabbit", "deer", "moose",
    "raccoon", "squirrel", "beaver", "lynx", "leopard", "cheetah", "jaguar",
    "hyena", "crocodile", "alligator", "iguana", "gecko", "cobra", "viper",
    "boa", "python", "tortoise",
]
_HABITAT_PREFIXES = [
    "savanna", "tundra", "forest", "desert", "ocean", "river", "mountain",
    "wetland", "grassland", "jungle", "reef", "swamp",
]
_COMPOUNDS = [
    "water", "salt", "sugar", "iron", "rust", "quartz", "limestone",
    "marble", "granite", "graphite", "diamond", "calcite", "pyrite",
    "gypsum", "talc", "feldspar", "obsidian", "basalt", "shale", "slate",
    "amber", "chalk", "clay", "coal", "bronze", "brass", "steel", "pewter",
    "solder", "amalgam",
]
_ELEMENTS = [
    "hydrogen", "oxygen", "carbon", "nitrogen", "sodium", "chlorine",
    "iron", "copper", "silicon", "calcium", "silver", "gold", "lead",
    "tin", "zinc", "sulfur", "phosphorus", "potassium", "magnesium",
    "aluminum",
]


def _build_a_corpus(n: int = 5000, seed: int = 0) -> list[tuple[str, str, str]]:
    """5K animals + habitats. Subjects are deterministic 'animal_i' -> a stable
    habitat assignment, so the same (s, lives_in, ?) question has one answer."""
    rng = random.Random(seed)
    corpus: list[tuple[str, str, str]] = []
    for i in range(n):
        animal = f"{_ANIMALS[i % len(_ANIMALS)]}_{i}"
        habitat = f"{_HABITAT_PREFIXES[rng.randrange(len(_HABITAT_PREFIXES))]}_{i % 7}"
        corpus.append((animal, "lives_in", habitat))
    return corpus


def _build_b_corpus(n: int = 5000, seed: int = 0) -> list[tuple[str, str, str]]:
    """5K chemistry + composition. Subjects deterministic 'compound_i' -> element."""
    rng = random.Random(seed + 1)
    corpus: list[tuple[str, str, str]] = []
    for i in range(n):
        compound = f"{_COMPOUNDS[i % len(_COMPOUNDS)]}_{i}"
        element = _ELEMENTS[rng.randrange(len(_ELEMENTS))]
        corpus.append((compound, "made_of", element))
    return corpus


def _is_correct(answer_text: str, citations: list, expected_object: str) -> bool:
    """A query is correct if the expected object appears in the model's
    answer text or in the cited fact chain."""
    target = expected_object.lower()
    if target in (answer_text or "").lower():
        return True
    return target in str(citations).lower()


@dataclass
class N1Result:
    benchmark: str
    n_a_facts: int
    n_b_facts: int
    n_probe: int
    initial_recall: float
    post_b_recall: float
    retention: float
    pass_threshold: float
    kill_threshold: float
    overall_pass: bool
    kill_triggered: bool
    seed: int
    per_query: list[dict] = field(default_factory=list)


def run_benchmark(
    n_a: int = 5000,
    n_b: int = 5000,
    n_probe: int = 200,
    dim: int = 4096,
    num_shards: int = 32,
    seed: int = 0,
    record_per_query: bool = False,
) -> N1Result:
    """Run the A -> B -> A retention benchmark.

    Returns a result with retention = post_b_recall / initial_recall.
    """
    a_facts = _build_a_corpus(n=n_a, seed=seed)
    b_facts = _build_b_corpus(n=n_b, seed=seed)

    rng = random.Random(seed)
    probe_idx = rng.sample(range(len(a_facts)), min(n_probe, len(a_facts)))
    probe_facts = [a_facts[i] for i in probe_idx]

    agent = ConsciousAgent(dim=dim, num_shards=num_shards, seed=seed)

    for s, r, o in a_facts:
        agent.tell(s, r, o)

    initial_hits = 0
    initial_per_query: list[dict] = []
    for s, r, expected in probe_facts:
        ans = agent.ask(s, r)
        ok = _is_correct(ans.text, ans.citations, expected)
        if ok:
            initial_hits += 1
        if record_per_query:
            initial_per_query.append({
                "phase": "initial", "subject": s, "relation": r,
                "expected": expected, "correct": ok,
                "inference_source": ans.inference_source,
            })

    for s, r, o in b_facts:
        agent.tell(s, r, o)

    post_hits = 0
    post_per_query: list[dict] = []
    for s, r, expected in probe_facts:
        ans = agent.ask(s, r)
        ok = _is_correct(ans.text, ans.citations, expected)
        if ok:
            post_hits += 1
        if record_per_query:
            post_per_query.append({
                "phase": "post_b", "subject": s, "relation": r,
                "expected": expected, "correct": ok,
                "inference_source": ans.inference_source,
            })

    n = len(probe_facts)
    initial_recall = initial_hits / n if n else 0.0
    post_b_recall = post_hits / n if n else 0.0
    retention = (post_b_recall / initial_recall) if initial_recall > 0 else 1.0

    pass_threshold = 0.50
    kill_threshold = 0.40

    return N1Result(
        benchmark="N1_continual_retention",
        n_a_facts=len(a_facts),
        n_b_facts=len(b_facts),
        n_probe=n,
        initial_recall=initial_recall,
        post_b_recall=post_b_recall,
        retention=retention,
        pass_threshold=pass_threshold,
        kill_threshold=kill_threshold,
        overall_pass=(retention >= pass_threshold),
        kill_triggered=(retention < kill_threshold),
        seed=seed,
        per_query=(initial_per_query + post_per_query) if record_per_query else [],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N1 continual retention benchmark")
    parser.add_argument("--n-a", type=int, default=5000)
    parser.add_argument("--n-b", type=int, default=5000)
    parser.add_argument("--n-probe", type=int, default=200)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-per-query", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark(
        n_a=args.n_a, n_b=args.n_b, n_probe=args.n_probe,
        dim=args.dim, num_shards=args.num_shards, seed=args.seed,
        record_per_query=args.record_per_query,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
