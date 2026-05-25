# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Benchmark RAIN-Net retrieval vs baselines on a synthetic QA set.

We want a measurable claim for the Series A pitch: "RAIN-Net retrieves
the correct fact at top-1 with X% accuracy on a synthetic 500-fact
benchmark, vs Y% for naive lexical baselines."

This script:
    1. Generates a synthetic KB of (fact, paraphrased_query) pairs
    2. Ingests facts into RAIN-Net (semantic memory)
    3. Runs each paraphrased query and measures whether the right fact
       comes back at top-1 / top-3 / top-5
    4. Repeats with two baselines:
       a. Naive lexical (Jaccard over word sets) — proxy for "would a
          dumb retriever work?"
       b. Random — proxy for floor
    5. Prints a comparison table

Run:
    python scripts/benchmark_rain_net.py --n-facts 500 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys

from rain.core.hv_substrate import similarity
from rain.core.rain_net import RainNet, RainNetConfig


# -------------------- synthetic dataset --------------------


# Fact templates and paraphrase templates. Pairs are designed to be
# realistic: the paraphrase is NOT lexical-identical, so a dumb retriever
# can't trivially win.
TEMPLATES = [
    (
        "Apollo mission {n} landed astronauts on the lunar surface in {year}.",
        "When did Apollo {n} land on the Moon?",
    ),
    (
        "The chemical symbol for {element} is {sym}.",
        "What element does {sym} stand for?",
    ),
    (
        "The capital of {country} is {city}.",
        "Which city is the capital of {country}?",
    ),
    (
        "{lang} was created by {person} in {year}.",
        "Who created the {lang} programming language?",
    ),
    (
        "{book} was written by {author} in {year}.",
        "Who is the author of {book}?",
    ),
]


def make_qa_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """Generate n (fact, query, expected_id) triples."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    apollos = [(11, 1969), (12, 1969), (14, 1971), (15, 1971), (16, 1972), (17, 1972)]
    elements = [
        ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"),
        ("Carbon", "C"), ("Nitrogen", "N"), ("Oxygen", "O"),
        ("Sodium", "Na"), ("Magnesium", "Mg"), ("Iron", "Fe"), ("Gold", "Au"),
        ("Silver", "Ag"), ("Copper", "Cu"), ("Zinc", "Zn"), ("Lead", "Pb"),
        ("Mercury", "Hg"), ("Platinum", "Pt"), ("Uranium", "U"),
        ("Calcium", "Ca"), ("Potassium", "K"), ("Argon", "Ar"),
    ]
    countries = [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("India", "New Delhi"),
        ("China", "Beijing"), ("Mexico", "Mexico City"), ("Norway", "Oslo"),
        ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Egypt", "Cairo"),
        ("Iceland", "Reykjavik"), ("Poland", "Warsaw"), ("Greece", "Athens"),
        ("Portugal", "Lisbon"), ("Turkey", "Ankara"),
    ]
    langs = [
        ("Python", "Guido van Rossum", 1991), ("Ruby", "Yukihiro Matsumoto", 1995),
        ("JavaScript", "Brendan Eich", 1995), ("Rust", "Graydon Hoare", 2010),
        ("Go", "Robert Griesemer", 2009), ("Lua", "Roberto Ierusalimschy", 1993),
        ("Haskell", "Simon Peyton Jones", 1990), ("Erlang", "Joe Armstrong", 1986),
        ("OCaml", "Xavier Leroy", 1996), ("Scala", "Martin Odersky", 2004),
    ]
    books = [
        ("Pride and Prejudice", "Jane Austen", 1813),
        ("Moby Dick", "Herman Melville", 1851),
        ("War and Peace", "Leo Tolstoy", 1869),
        ("The Great Gatsby", "F. Scott Fitzgerald", 1925),
        ("1984", "George Orwell", 1949),
        ("Brave New World", "Aldous Huxley", 1932),
        ("Catch-22", "Joseph Heller", 1961),
        ("Beloved", "Toni Morrison", 1987),
        ("Dune", "Frank Herbert", 1965),
        ("Slaughterhouse-Five", "Kurt Vonnegut", 1969),
    ]
    pools = [
        ("apollo", apollos, lambda n, y: (
            TEMPLATES[0][0].format(n=n, year=y),
            TEMPLATES[0][1].format(n=n),
        )),
        ("element", elements, lambda e, s: (
            TEMPLATES[1][0].format(element=e, sym=s),
            TEMPLATES[1][1].format(sym=s),
        )),
        ("country", countries, lambda c, city: (
            TEMPLATES[2][0].format(country=c, city=city),
            TEMPLATES[2][1].format(country=c),
        )),
        ("lang", langs, lambda l, p, y: (
            TEMPLATES[3][0].format(lang=l, person=p, year=y),
            TEMPLATES[3][1].format(lang=l),
        )),
        ("book", books, lambda b, a, y: (
            TEMPLATES[4][0].format(book=b, author=a, year=y),
            TEMPLATES[4][1].format(book=b),
        )),
    ]
    while len(out) < n:
        pool_name, items, mk = pools[rng.randint(0, len(pools) - 1)]
        args = rng.choice(items)
        fact, q = mk(*args)
        fid = f"{pool_name}_{len(out)}"
        out.append((fact, q, fid))
    return out


# -------------------- baselines --------------------


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_retrieve(facts: list[tuple[str, str]], query: str, top_k: int) -> list[str]:
    """Naive lexical baseline: rank facts by Jaccard word overlap."""
    scored = [(jaccard(query, f_text), fid) for f_text, fid in facts]
    scored.sort(key=lambda x: -x[0])
    return [fid for _s, fid in scored[:top_k]]


def random_retrieve(facts: list[tuple[str, str]], top_k: int, rng: random.Random) -> list[str]:
    sample = facts[:]
    rng.shuffle(sample)
    return [fid for _, fid in sample[:top_k]]


def rain_retrieve(net: RainNet, query: str, top_k: int) -> list[str]:
    report = net.answer(query)
    return [f.fact_id for f in report.cited_facts[:top_k]]


# -------------------- bench loop --------------------


def evaluate(
    retriever_name: str,
    retrieve_fn,
    pairs: list[tuple[str, str, str]],
    top_ks: list[int],
) -> dict[str, float]:
    """Returns {f'top{k}': accuracy} per k."""
    correct_at_k = {k: 0 for k in top_ks}
    for _fact_text, query, fid in pairs:
        topk_max = max(top_ks)
        results = retrieve_fn(query, topk_max)
        for k in top_ks:
            if fid in results[:k]:
                correct_at_k[k] += 1
    total = len(pairs)
    return {f"top{k}": correct_at_k[k] / total for k in top_ks}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-facts", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=10_000)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--top-ks", default="1,3,5")
    args = p.parse_args(argv)

    top_ks = [int(x) for x in args.top_ks.split(",")]
    pairs = make_qa_pairs(args.n_facts, seed=args.seed)
    print(f"Generated {len(pairs)} (fact, query, id) pairs.")

    # Build RainNet, ingest facts.
    net = RainNet(
        config=RainNetConfig(
            dim=args.dim,
            n_candidates=args.candidates,
            semantic_top_k=max(top_ks),
        )
    )
    for fact_text, _q, fid in pairs:
        net.ingest_fact(fact_text, fact_id=fid)
    print(f"Ingested {len(pairs)} facts into RAIN-Net semantic memory.")

    # The facts-only list used by baseline retrievers (no IDs leaked
    # in the text so they have to match by content).
    facts_for_baseline = [(f, fid) for f, _q, fid in pairs]
    rng = random.Random(args.seed)

    print()
    print(f"{'Retriever':<25} " + " ".join(f"top-{k:<3}" for k in top_ks))
    print("-" * (25 + 8 * len(top_ks)))

    # Random floor
    random_results = evaluate(
        "random",
        lambda q, k: random_retrieve(facts_for_baseline, k, rng),
        pairs,
        top_ks,
    )
    print(f"{'Random (floor)':<25} " + " ".join(f"{random_results[f'top{k}']:.3f}  " for k in top_ks))

    # Lexical Jaccard baseline
    lex_results = evaluate(
        "lexical_jaccard",
        lambda q, k: lexical_retrieve(facts_for_baseline, q, k),
        pairs,
        top_ks,
    )
    print(f"{'Lexical (Jaccard)':<25} " + " ".join(f"{lex_results[f'top{k}']:.3f}  " for k in top_ks))

    # RAIN-Net (HV semantic similarity)
    rain_results = evaluate(
        "rain_net",
        lambda q, k: rain_retrieve(net, q, k),
        pairs,
        top_ks,
    )
    print(f"{'RAIN-Net (HV)':<25} " + " ".join(f"{rain_results[f'top{k}']:.3f}  " for k in top_ks))

    # Print summary.
    print()
    print("Notes:")
    print(f"  * KB size: {len(pairs)} facts")
    print(f"  * HV dim:  {args.dim}")
    print("  * Lexical Jaccard uses word-set overlap (no embeddings)")
    print("  * RAIN-Net uses n-gram text encoder + semantic memory")
    print("  * No training was needed for RAIN-Net retrieval (n-gram is deterministic)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
