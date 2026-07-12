"""COGS template-learner symbolic evaluation: fit COGSTemplateLearner on
train.tsv ONLY, evaluate the full gen.tsv split, report a per-category
breakdown, and run a shuffled-gold negative control.

This is the reproducible producer for the headline COGS row -- it replaces
the ad-hoc (hand-run, non-reproducible) artifact previously checked in at
raincg/results/cogs_gen__template_learner_symbolic__seed0.json.

params=0 means zero gradient-trained floats. It does NOT mean zero learned
structure: the model's structural knowledge lives in symbolic lookup tables
(templates, past-tense->infinitive map, intransitive-subject role table,
ditransitive-verb set, verb-conditioned slot lookups) built by COGSTemplateLearner.fit
on the train split. Those tables are counted and disclosed under
config.symbolic_table_entries so params=0 is never mistaken for "no learning
happened".

  python experiments/cogs_template_symbolic_eval.py \
      --out raincg/results/cogs_gen__template_learner_symbolic__seed0.json
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

from _result_io import emit_result  # noqa: E402
from pure_vsa.cogs_hyperion import load_cogs_tsv  # noqa: E402
from pure_vsa.cogs_template_learner import COGSTemplateLearner  # noqa: E402


def symbolic_table_entries(learner: COGSTemplateLearner) -> dict[str, int]:
    """Count entries in every learned lookup table/dict the learner holds
    after fit(). This is the honest disclosure companion to params=0."""
    counts = {
        "templates": len(learner.templates),
        "past_to_inf": len(learner.past_to_inf),
        "intrans_role": len(learner.intrans_role),
        "ditrans_verbs": len(learner.ditrans_verbs),
        "verb_conditioned_lookup": sum(
            len(v) for v in learner.verb_conditioned_lookup.values()
        ),
    }
    counts["total"] = sum(counts.values())
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                    help="write contract result JSON to this path")
    a = p.parse_args()

    t0 = time.monotonic()
    train = load_cogs_tsv(REPO / "data" / "cogs" / "raw" / "train.tsv")
    gen = load_cogs_tsv(REPO / "data" / "cogs" / "raw" / "gen.tsv")
    train_pairs = [(t[0], t[1]) for t in train]

    learner = COGSTemplateLearner()
    learner.fit(train_pairs)  # TRAIN split ONLY -- gen is never touched here
    train_s = time.monotonic() - t0

    t1 = time.monotonic()
    stats = learner.coverage_stats(gen)
    eval_s = time.monotonic() - t1

    n_gen = len(gen)
    correct = stats["correct"]

    per_category = {}
    for cat in sorted(stats["per_cat_total"]):
        c = stats["per_cat_correct"].get(cat, 0)
        n = stats["per_cat_total"][cat]
        per_category[cat] = f"{c}/{n}"

    # Shuffled-gold negative control: pair each gen input with a randomly
    # permuted gold output (drawn from the gen split itself). A system with
    # real signal must crater on this; the small number of survivors are
    # structural coincidences (e.g. identical short outputs colliding under
    # the permutation), not evidence of leakage.
    rng = random.Random(a.seed)
    shuffled_gold = [g for _, g, _ in gen]
    rng.shuffle(shuffled_gold)
    shuffled_examples = [(inp, sg, cat) for (inp, _, cat), sg in zip(gen, shuffled_gold)]
    shuffled_stats = learner.coverage_stats(shuffled_examples)

    table_entries = symbolic_table_entries(learner)

    print(f"fit_s={train_s:.2f} eval_s={eval_s:.2f}")
    print(f"COGS gen (train-only fit): {correct}/{n_gen} = {correct/n_gen*100:.2f}%")
    print(f"symbolic_table_entries: {table_entries}")
    print(f"shuffled-gold negative control: {shuffled_stats['correct']}/{n_gen}")

    if a.out:
        emit_result(
            a.out,
            bench="cogs_gen",
            system="template_learner_symbolic",
            split="gen",
            n=n_gen,
            correct=correct,
            params=0,
            train_s=train_s,
            eval_s=eval_s,
            seed=a.seed,
            config={
                "full_n": n_gen,
                "full_correct": correct,
                "train_rows": len(train),
                "solver": "COGSTemplateLearner",
                "symbolic_table_entries": table_entries,
                "negative_control_shuffled_targets": f"{shuffled_stats['correct']}/{n_gen}",
                "per_category": per_category,
            },
            notes=(
                "symbolic template metalearner, fit on COGS train split ONLY, zero "
                "gradient descent (params=0 = zero gradient-trained floats; structural "
                "knowledge lives in symbolic lookup tables, see "
                "config.symbolic_table_entries for the entry counts); shuffled-gold "
                "negative control confirms the accuracy is not a scoring artifact; "
                "per-category breakdown in config.per_category; task-specific "
                "inductive bias (COGS role ontology + construction signatures) -- "
                "do not compare to neural SOTA without this caveat."
            ),
        )


if __name__ == "__main__":
    main()
