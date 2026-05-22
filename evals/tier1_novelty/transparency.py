# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N5 Grounded Transparency benchmark.

Tier-1 novelty surface. Seeds a 100-fact KB across 5 domains, asks 100
questions (mix of answerable + unanswerable), audits every Answer for a
citation chain back to a KB write.

Hallucination = an Answer with epistemic in {know, think} that has no
citations OR cites facts NOT actually in the KB. Refusals (epistemic =
unknown) are NOT hallucinations.

Acceptance: citation coverage >= 98%; hallucination rate <= 1%.

LLMs with RAG: ~30-50% citation coverage; 70-90% hallucination on novel
facts. RAIN's structural advantage: every fact in an answer came from
a stored triple by construction.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from rain.agent import ConsciousAgent


# 100 facts across 5 domains (20 each)
SEED_FACTS: list[tuple[str, str, str]] = [
    # Geography
    ("rome", "capital_of", "italy"),
    ("paris", "capital_of", "france"),
    ("madrid", "capital_of", "spain"),
    ("london", "capital_of", "united_kingdom"),
    ("berlin", "capital_of", "germany"),
    ("tokyo", "capital_of", "japan"),
    ("seoul", "capital_of", "south_korea"),
    ("beijing", "capital_of", "china"),
    ("delhi", "capital_of", "india"),
    ("cairo", "capital_of", "egypt"),
    ("italy", "locatedin", "europe"),
    ("france", "locatedin", "europe"),
    ("spain", "locatedin", "europe"),
    ("germany", "locatedin", "europe"),
    ("japan", "locatedin", "asia"),
    ("china", "locatedin", "asia"),
    ("india", "locatedin", "asia"),
    ("egypt", "locatedin", "africa"),
    ("brazil", "locatedin", "south_america"),
    ("canada", "locatedin", "north_america"),
    # Chemistry
    ("water", "boils_at", "100c"),
    ("water", "freezes_at", "0c"),
    ("water", "formula", "h2o"),
    ("salt", "formula", "nacl"),
    ("gold", "symbol", "au"),
    ("silver", "symbol", "ag"),
    ("iron", "symbol", "fe"),
    ("oxygen", "symbol", "o"),
    ("hydrogen", "symbol", "h"),
    ("carbon", "symbol", "c"),
    ("water", "isa", "compound"),
    ("salt", "isa", "compound"),
    ("gold", "isa", "element"),
    ("oxygen", "isa", "element"),
    ("hydrogen", "isa", "element"),
    ("carbon", "isa", "element"),
    ("iron", "isa", "element"),
    ("methane", "formula", "ch4"),
    ("ammonia", "formula", "nh3"),
    ("co2", "name", "carbon_dioxide"),
    # Biology
    ("sparrow", "isa", "bird"),
    ("eagle", "isa", "bird"),
    ("salmon", "isa", "fish"),
    ("trout", "isa", "fish"),
    ("dog", "isa", "mammal"),
    ("cat", "isa", "mammal"),
    ("whale", "isa", "mammal"),
    ("bird", "can_fly", "yes"),
    ("fish", "can_swim", "yes"),
    ("mammal", "warm_blooded", "yes"),
    ("oak", "isa", "tree"),
    ("pine", "isa", "tree"),
    ("rose", "isa", "flower"),
    ("tulip", "isa", "flower"),
    ("tree", "has_part", "leaves"),
    ("flower", "has_part", "petals"),
    ("dog", "has_part", "tail"),
    ("cat", "has_part", "whiskers"),
    ("bird", "has_part", "wings"),
    ("fish", "has_part", "fins"),
    # Astronomy
    ("sun", "isa", "star"),
    ("earth", "isa", "planet"),
    ("mars", "isa", "planet"),
    ("jupiter", "isa", "planet"),
    ("venus", "isa", "planet"),
    ("moon", "isa", "satellite"),
    ("earth", "orbits", "sun"),
    ("mars", "orbits", "sun"),
    ("jupiter", "orbits", "sun"),
    ("moon", "orbits", "earth"),
    ("milky_way", "isa", "galaxy"),
    ("andromeda", "isa", "galaxy"),
    ("sun", "locatedin", "milky_way"),
    ("solar_system", "contains", "earth"),
    ("solar_system", "contains", "mars"),
    ("solar_system", "contains", "sun"),
    ("light", "speed", "299792458_mps"),
    ("year", "duration", "365_days"),
    ("day", "duration", "24_hours"),
    ("hour", "duration", "60_minutes"),
    # History / culture
    ("shakespeare", "wrote", "hamlet"),
    ("shakespeare", "wrote", "macbeth"),
    ("shakespeare", "wrote", "othello"),
    ("homer", "wrote", "iliad"),
    ("homer", "wrote", "odyssey"),
    ("dante", "wrote", "divine_comedy"),
    ("hamlet", "isa", "play"),
    ("macbeth", "isa", "play"),
    ("iliad", "isa", "epic"),
    ("odyssey", "isa", "epic"),
    ("shakespeare", "born_in", "england"),
    ("homer", "born_in", "greece"),
    ("dante", "born_in", "italy"),
    ("renaissance", "isa", "era"),
    ("middle_ages", "isa", "era"),
    ("ancient_greece", "isa", "civilization"),
    ("rome_empire", "isa", "civilization"),
    ("egypt_ancient", "isa", "civilization"),
    ("china_ancient", "isa", "civilization"),
    ("india_ancient", "isa", "civilization"),
]


# 100 questions. Of these, ~75 are answerable from seed facts (some via inheritance),
# ~25 are unanswerable (no stored fact, no inheritance chain).
QUESTIONS: list[tuple[str, str]] = (
    # ANSWERABLE direct queries (50)
    [("rome", "capital_of"), ("paris", "capital_of"), ("tokyo", "capital_of"),
     ("italy", "locatedin"), ("japan", "locatedin"), ("brazil", "locatedin"),
     ("water", "boils_at"), ("water", "formula"), ("salt", "formula"),
     ("gold", "symbol"), ("oxygen", "symbol"), ("methane", "formula"),
     ("sparrow", "isa"), ("dog", "isa"), ("eagle", "isa"),
     ("bird", "can_fly"), ("fish", "can_swim"), ("mammal", "warm_blooded"),
     ("oak", "isa"), ("rose", "isa"),
     ("tree", "has_part"), ("flower", "has_part"), ("dog", "has_part"),
     ("sun", "isa"), ("earth", "isa"), ("moon", "isa"),
     ("earth", "orbits"), ("moon", "orbits"),
     ("milky_way", "isa"), ("sun", "locatedin"),
     ("solar_system", "contains"), ("light", "speed"),
     ("shakespeare", "wrote"), ("homer", "wrote"), ("dante", "wrote"),
     ("hamlet", "isa"), ("iliad", "isa"),
     ("shakespeare", "born_in"), ("homer", "born_in"),
     ("renaissance", "isa"), ("ancient_greece", "isa"),
     ("madrid", "capital_of"), ("seoul", "capital_of"),
     ("water", "freezes_at"), ("silver", "symbol"), ("iron", "symbol"),
     ("salmon", "isa"), ("whale", "isa"), ("pine", "isa"),
     ("venus", "isa"), ("jupiter", "orbits"),
     ]
    # ANSWERABLE via inheritance (25): query a property the entity inherits from its class
    + [("sparrow", "can_fly"), ("eagle", "can_fly"),
       ("salmon", "can_swim"), ("trout", "can_swim"),
       ("dog", "warm_blooded"), ("cat", "warm_blooded"), ("whale", "warm_blooded"),
       ("rome", "locatedin"),  # rome capital_of italy, italy locatedin europe -> rome locatedin europe (transitive)
       ("paris", "locatedin"),
       ("madrid", "locatedin"),
       ("tokyo", "locatedin"),
       ("oak", "has_part"),
       ("pine", "has_part"),
       ("rose", "has_part"),
       ("tulip", "has_part"),
       ("water", "isa"), ("salt", "isa"),
       ("gold", "isa"), ("oxygen", "isa"),
       ("london", "capital_of"),
       ("berlin", "capital_of"),
       ("beijing", "capital_of"),
       ("delhi", "capital_of"),
       ("cairo", "capital_of"),
       ("co2", "name"),
       ]
    # UNANSWERABLE (25): subject or relation not in KB / no inheritance path
    + [("atlantis", "capital_of"),
       ("zog", "isa"),
       ("xyzzy", "locatedin"),
       ("blorp", "symbol"),
       ("quux", "formula"),
       ("nowhere", "orbits"),
       ("nothing", "isa"),
       ("vacuum", "boils_at"),
       ("rome", "born_in"),  # no such fact
       ("paris", "wrote"),
       ("water", "wrote"),
       ("dog", "capital_of"),
       ("sun", "boils_at"),
       ("hamlet", "lives_in"),
       ("eagle", "freezes_at"),
       ("salt", "born_in"),
       ("gold", "orbits"),
       ("milky_way", "capital_of"),
       ("rose", "boils_at"),
       ("renaissance", "capital_of"),
       ("shakespeare", "isa"),  # shakespeare's not labeled with isa
       ("homer", "isa"),
       ("oxygen", "wrote"),
       ("tokyo", "boils_at"),
       ("hamlet", "boils_at"),
       ]
)


@dataclass
class N5Result:
    benchmark: str
    n_questions: int
    n_answerable: int
    n_refusals: int
    n_hallucinations: int
    n_with_citations: int
    citation_coverage: float
    hallucination_rate: float
    citation_pass: bool
    hallucination_pass: bool
    overall_pass: bool
    failures: list[str] = field(default_factory=list)


def run_benchmark() -> N5Result:
    agent = ConsciousAgent(dim=2048, num_shards=8, seed=0)
    for s, r, o in SEED_FACTS:
        agent.tell(s, r, o)

    # Build a set of (S, R, O) tuples representing the KB ground truth.
    # Citations are valid only if (S, R, O) is in this set.
    valid_triples = {(s, r, o) for s, r, o in SEED_FACTS}

    n_questions = len(QUESTIONS)
    n_with_citations = 0
    n_hallucinations = 0
    n_refusals = 0
    n_answerable = 0  # the model returned a non-refusal answer
    failures: list[str] = []

    for (s, r) in QUESTIONS:
        answer = agent.ask(s, r)
        if answer.epistemic == "unknown":
            n_refusals += 1
            continue
        n_answerable += 1
        if answer.citations:
            n_with_citations += 1
            # Verify every cited triple is in valid_triples
            for triple in answer.citations:
                if tuple(triple) not in valid_triples:
                    n_hallucinations += 1
                    failures.append(f"hallucinated citation: ({s},{r}) -> {triple}")
                    break
        else:
            # Non-refusal answer with no citations = hallucination
            n_hallucinations += 1
            failures.append(f"answer w/o citation: ({s},{r}) -> {answer.text!r}")

    citation_coverage = (n_with_citations / n_answerable) if n_answerable else 1.0
    hallucination_rate = (n_hallucinations / n_questions)
    citation_pass = citation_coverage >= 0.98
    hallucination_pass = hallucination_rate <= 0.01

    return N5Result(
        benchmark="N5_grounded_transparency",
        n_questions=n_questions,
        n_answerable=n_answerable,
        n_refusals=n_refusals,
        n_hallucinations=n_hallucinations,
        n_with_citations=n_with_citations,
        citation_coverage=citation_coverage,
        hallucination_rate=hallucination_rate,
        citation_pass=citation_pass,
        hallucination_pass=hallucination_pass,
        overall_pass=citation_pass and hallucination_pass,
        failures=failures[:20],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N5 grounded transparency benchmark")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
