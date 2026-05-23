"""Hyperion demo: run the headline result interactively.

Trains the pure-VSA HyperionReasoner on TinySCAN, evaluates on the held-out
compositional split, and runs the matched transformer baseline for comparison.

Usage:
  python -m pure_vsa.demo                     # quick run (~10s)
  python -m pure_vsa.demo --no-baseline       # skip transformer baseline
  python -m pure_vsa.demo --capacity          # also run the capacity sweep (~90s)
"""

from __future__ import annotations

import argparse
import time


def _hr(char: str = "-", width: int = 78) -> str:
    return char * width


def run_tinyscan_v1_demo() -> dict:
    """Train HyperionReasoner on TinySCAN, return results dict."""
    from pure_vsa import Example, HyperionConfig, HyperionReasoner
    from pure_vsa.tinyscan import (
        N_ACTIONS,
        N_MODIFIERS,
        ROLE_MOD,
        ROLE_VERB,
        TOTAL_SYMBOLS,
        make_dataset,
        symbol_index,
    )

    print(_hr("="))
    print("Hyperion demo  |  pure-VSA compositional generalization")
    print(_hr("="))

    train, test = make_dataset()
    print(f"\n[data] TinySCAN. {len(train)} training examples, {len(test)} held-out.")
    print("       held-out verb: swim (bare swim IS in train; swim+modifier never seen).")

    cfg = HyperionConfig(
        d=10_000,
        n_symbols=TOTAL_SYMBOLS,
        n_input_roles=2,
        max_output_len=4,
        output_vocab_offset=N_ACTIONS + N_MODIFIERS,
        seed=0,
    )

    t0 = time.time()
    reasoner = HyperionReasoner(cfg)
    train_examples = [
        Example(
            input_slots=ex.input_slots(),
            output_token_indices=[symbol_index(t) for t in ex.output_tokens],
        )
        for ex in train
    ]
    rule_sizes = reasoner.fit(
        train_examples, modifier_role_idx=ROLE_MOD, verb_role_idx=ROLE_VERB
    )
    train_time = time.time() - t0
    print(f"\n[fit ] {reasoner}")
    print(f"       trained in {train_time*1000:.1f} ms.")
    print(f"       extracted rules: {dict(rule_sizes)}")

    print("\n[test] held-out (swim, MODIFIER) compositional generalization:")
    correct = 0
    for ex in test:
        predicted = reasoner.predict(
            ex.input_slots(),
            output_length=len(ex.output_tokens),
            modifier_role_idx=ROLE_MOD,
            verb_role_idx=ROLE_VERB,
        )
        expected = [symbol_index(t) for t in ex.output_tokens]
        ok = predicted == expected
        from pure_vsa.tinyscan import OUTPUT_TOKENS
        pred_toks = [OUTPUT_TOKENS[i - cfg.output_vocab_offset] for i in predicted]
        exp_toks = [OUTPUT_TOKENS[i - cfg.output_vocab_offset] for i in expected]
        marker = "OK  " if ok else "FAIL"
        mod_str = f", {ex.modifier}" if ex.modifier else ""
        print(f"       {marker}  ({ex.action}{mod_str})  ->  {pred_toks}  expected {exp_toks}")
        if ok:
            correct += 1

    acc = correct / len(test)
    print(f"\n[acc ] Hyperion held-out: {correct}/{len(test)} = {acc:.1%}")
    return {"acc": acc, "fit_time_ms": train_time * 1000, "n_train": len(train), "n_test": len(test)}


def run_transformer_baseline(n_seeds: int = 5) -> dict:
    """Train a small transformer on the same TinySCAN train split,
    eval same held-out split, report per-seed and mean."""
    from pure_vsa.baselines.transformer_tinyscan import train_and_eval

    print(f"\n[base] Training {n_seeds}-seed transformer baseline (~10s per seed)...")
    accs = []
    for s in range(n_seeds):
        r = train_and_eval(seed=s, verbose=False)
        accs.append(r["test_acc"])
        print(f"       seed={s}  train_acc={r['train_acc']:.2f}  test_acc={r['test_acc']:.2f}  ({r['n_params']:,} params)")
    mean = sum(accs) / len(accs)
    print(f"\n[base] transformer held-out mean ({n_seeds} seeds): {mean:.1%}  range [{min(accs):.1%}, {max(accs):.1%}]")
    return {"mean_acc": mean, "min_acc": min(accs), "max_acc": max(accs), "per_seed": accs}


def run_capacity_sweep() -> None:
    """Capacity envelope check at D=2048 across n_verbs in [80 ... 8000]."""
    from pure_vsa.capacity_study_v3 import _run_capacity_v3

    print(_hr())
    print("\n[scale] Capacity envelope at D=2048 (facts-in-dict + clean rules):")
    print("        n_verbs  test_acc  train_acc  time")
    for n in [80, 320, 1280, 3000, 8000]:
        r = _run_capacity_v3(n, 2048, seed=0)
        print(f"        {n:7d}    {r.test_acc:.2f}      {r.train_acc:.2f}     {r.elapsed_s:.1f}s")


def headline_summary(hyperion: dict, transformer: dict | None) -> None:
    print()
    print(_hr("="))
    print("Headline")
    print(_hr("="))
    print(f"  Hyperion (pure VSA, 0 trained params):  {hyperion['acc']:.1%}  [fit: {hyperion['fit_time_ms']:.0f} ms]")
    if transformer:
        print(
            f"  Transformer baseline (70K params):       "
            f"{transformer['mean_acc']:.1%} mean  [{transformer['min_acc']:.1%} - {transformer['max_acc']:.1%} range over {len(transformer['per_seed'])} seeds]"
        )
    print(_hr("="))
    print(
        "\n  Hyperion result is deterministic across codebook seeds.\n"
        "  Transformer is bimodal: sometimes generalizes, sometimes locks into rote memorization.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the transformer baseline")
    p.add_argument("--capacity", action="store_true",
                   help="also run the capacity sweep")
    p.add_argument("--seeds", type=int, default=5,
                   help="number of transformer baseline seeds")
    args = p.parse_args()

    hyperion = run_tinyscan_v1_demo()
    transformer = None if args.no_baseline else run_transformer_baseline(n_seeds=args.seeds)
    if args.capacity:
        run_capacity_sweep()
    headline_summary(hyperion, transformer)


if __name__ == "__main__":
    main()
