"""Train the HVVerifierHead on judge-labeled candidate scoring.

The verifier head's job is to score candidate answer HVs and pick the
best one (test-time-compute sample-and-rank). Out of the box it is
untrained; this script collects training data and runs it.

Pipeline:
    1. Build a labeled dataset of (query, good_answer, bad_answer) triples
       - Good answer: facts that genuinely answer the query.
       - Bad answer: random unrelated facts.
    2. Encode each triple; produce two training pairs:
       (query_hv, good_hv, +1) and (query_hv, bad_hv, -1).
    3. Train the verifier via the online perceptron update.
    4. Evaluate on a held-out set: for each (query, good, bad), check
       that score(query, good) > score(query, bad).
    5. Persist the verifier weights and the eval metrics.

Run:
    python scripts/train_verifier_head.py --n-pairs 200 --eval-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hv_substrate import DEFAULT_DIM
from rain.core.verifier_head import HVVerifierHead


# A reusable set of (query, good_fact, bad_fact_pool) triples designed
# to cover several semantic domains so the verifier learns a domain-
# agnostic notion of "this fact answers this query."
QA_BUNDLES = [
    # (query, good_answer, list_of_bad_candidates)
    (
        "When did Apollo 11 land on the Moon?",
        "Apollo 11 landed on the Moon on July 20, 1969.",
        [
            "Python was created by Guido van Rossum in 1991.",
            "The chemical symbol for Iron is Fe.",
            "Paris is the capital of France.",
            "Pride and Prejudice was written by Jane Austen.",
        ],
    ),
    (
        "What is the chemical symbol for Iron?",
        "The chemical symbol for Iron is Fe.",
        [
            "The capital of Japan is Tokyo.",
            "Rust was created by Graydon Hoare in 2010.",
            "Moby Dick was written by Herman Melville.",
            "Apollo 14 landed on the Moon in 1971.",
        ],
    ),
    (
        "Which city is the capital of Germany?",
        "The capital of Germany is Berlin.",
        [
            "JavaScript was created by Brendan Eich in 1995.",
            "The chemical symbol for Gold is Au.",
            "1984 was written by George Orwell.",
            "Apollo 17 landed on the Moon in 1972.",
        ],
    ),
    (
        "Who created the Python programming language?",
        "Python was created by Guido van Rossum in 1991.",
        [
            "The capital of Brazil is Brasilia.",
            "The chemical symbol for Hydrogen is H.",
            "War and Peace was written by Leo Tolstoy.",
            "Apollo 16 landed on the Moon in 1972.",
        ],
    ),
    (
        "Who is the author of Dune?",
        "Dune was written by Frank Herbert in 1965.",
        [
            "The capital of Mexico is Mexico City.",
            "The chemical symbol for Silver is Ag.",
            "Ruby was created by Yukihiro Matsumoto in 1995.",
            "Apollo 12 landed on the Moon in 1969.",
        ],
    ),
    (
        "When was JavaScript created?",
        "JavaScript was created by Brendan Eich in 1995.",
        [
            "The capital of Spain is Madrid.",
            "The chemical symbol for Helium is He.",
            "The Great Gatsby was written by F. Scott Fitzgerald.",
            "Apollo 15 landed on the Moon in 1971.",
        ],
    ),
    (
        "What element does Au stand for?",
        "The chemical symbol for Gold is Au.",
        [
            "The capital of Norway is Oslo.",
            "Lua was created by Roberto Ierusalimschy in 1993.",
            "Brave New World was written by Aldous Huxley.",
            "Apollo 11 landed on the Moon in 1969.",
        ],
    ),
    (
        "Who wrote 1984?",
        "1984 was written by George Orwell in 1949.",
        [
            "The capital of Italy is Rome.",
            "The chemical symbol for Carbon is C.",
            "Go was created by Robert Griesemer in 2009.",
            "Apollo 17 landed on the Moon in 1972.",
        ],
    ),
]


def build_dataset(n_pairs: int, seed: int) -> list[tuple[str, str, str]]:
    """Build n_pairs triples of (query, good, bad).

    Each triple is (query, ground-truth answer, distractor answer).
    For training we'll convert each triple to two perceptron updates.
    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str, str]] = []
    while len(pairs) < n_pairs:
        q, good, bads = rng.choice(QA_BUNDLES)
        bad = rng.choice(bads)
        pairs.append((q, good, bad))
    return pairs


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--dim", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/checkpoints/verifier_head_v0.npz")
    p.add_argument("--metrics-out", default="data/checkpoints/verifier_head_v0_metrics.json")
    args = p.parse_args(argv)

    print(f"=== Verifier head training ===")
    print(f"  n_pairs:   {args.n_pairs}")
    print(f"  eval frac: {args.eval_frac}")
    print(f"  dim:       {args.dim}")
    print(f"  lr:        {args.lr}")
    print(f"  epochs:    {args.epochs}")
    print()

    enc = EncoderBank(dim=args.dim)
    verifier = HVVerifierHead(dim=args.dim, learning_rate=args.lr)

    # Build dataset.
    pairs = build_dataset(args.n_pairs, args.seed)
    n_eval = int(len(pairs) * args.eval_frac)
    n_train = len(pairs) - n_eval
    train, eval_set = pairs[:n_train], pairs[n_train:]
    print(f"Built {len(pairs)} pairs ({n_train} train, {n_eval} eval).")

    # Eval BEFORE training (random init).
    def evaluate(verif, ds: list[tuple[str, str, str]]) -> dict[str, float]:
        good_scores = []
        bad_scores = []
        correct = 0
        for q, good, bad in ds:
            q_hv = enc.encode("text", q)
            g_hv = enc.encode("text", good)
            b_hv = enc.encode("text", bad)
            s_good = verif.score(q_hv, g_hv)
            s_bad = verif.score(q_hv, b_hv)
            good_scores.append(s_good)
            bad_scores.append(s_bad)
            if s_good > s_bad:
                correct += 1
        return {
            "accuracy": correct / len(ds),
            "mean_good_score": float(np.mean(good_scores)),
            "mean_bad_score": float(np.mean(bad_scores)),
            "score_gap": float(np.mean(good_scores) - np.mean(bad_scores)),
        }

    before = evaluate(verifier, eval_set)
    print(f"BEFORE training: acc={before['accuracy']:.3f} gap={before['score_gap']:+.4f}")

    # Train across epochs (multiple passes).
    for epoch in range(args.epochs):
        # Shuffle each epoch.
        epoch_pairs = train[:]
        random.Random(args.seed + epoch).shuffle(epoch_pairs)
        for q, good, bad in epoch_pairs:
            q_hv = enc.encode("text", q)
            g_hv = enc.encode("text", good)
            b_hv = enc.encode("text", bad)
            verifier.update(q_hv, g_hv, +1)
            verifier.update(q_hv, b_hv, -1)
        mid = evaluate(verifier, eval_set)
        print(
            f"epoch {epoch+1}/{args.epochs}: acc={mid['accuracy']:.3f} "
            f"gap={mid['score_gap']:+.4f} updates={verifier.n_updates}"
        )

    after = evaluate(verifier, eval_set)
    print()
    print(f"AFTER  training: acc={after['accuracy']:.3f} gap={after['score_gap']:+.4f}")

    # Persist the weights + metrics.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, w=verifier._w, n_updates=np.array(verifier.n_updates))
    print(f"Saved verifier head to {args.out}")

    metrics = {
        "n_pairs": args.n_pairs,
        "n_train": n_train,
        "n_eval": n_eval,
        "dim": args.dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "before_eval": before,
        "after_eval": after,
        "n_updates": verifier.n_updates,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
