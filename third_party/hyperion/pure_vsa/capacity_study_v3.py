"""Capacity study v3: separate facts from rules.

Facts (bare-verb -> output-symbol mappings) are stored in a Python dict.
Rules are still pure VSA (pattern + residual extracted from training outputs).

Architectural rationale: bundled VSA memory is the right tool for queries with
structural variation (e.g. retrieve based on a partial query, or compose on the
fly). For primitive (key -> value) facts that are *complete* at training time,
direct table lookup avoids the bundle's cross-talk noise entirely.

The compositional generalization claim is unchanged -- composition still happens
via VSA algebra over the rule HVs. Only the substrate for primitive fact storage
moves out of the bundle.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from pure_vsa.capacity_study import (
    MODIFIERS,
    N_MAX_OUT,
    N_ROLES,
    ROLE_VERB,
    _build_dataset,
    _encode_output,
)
from pure_vsa.reasoner import PureVSAReasoner
from vsa_core import bind, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


@dataclass
class CapacityResultV3:
    n_verbs: int
    d: int
    train_size: int
    test_size: int
    train_acc: float
    test_acc: float
    elapsed_s: float


def _run_capacity_v3(n_verbs: int, d: int, seed: int = 0) -> CapacityResultV3:
    torch.manual_seed(seed)
    t0 = time.time()

    held_out_verb_idx = 0
    train, test, total_symbols, LEFT_idx, RIGHT_idx, out_offset = _build_dataset(
        n_verbs, held_out_verb_idx
    )

    symbols = Codebook(total_symbols, d, seed=seed)
    roles = Codebook(N_ROLES, d, seed=seed + 1)
    reasoner = PureVSAReasoner(d=d, symbol_codebook=symbols, role_codebook=roles)

    from vsa_core import permute as vsa_permute  # noqa: PLC0415
    base = roles[ROLE_VERB]
    output_role_hvs = torch.stack(
        [vsa_permute(base, shift=i) for i in range(N_MAX_OUT)]
    )

    # FACT STORE: dict of bare-verb input -> output symbol index. Populated at
    # training time. Constant-time lookup at test time; zero cross-talk.
    fact_store: dict[int, int] = {}
    for v_idx, m_idx, output_tokens in train:
        if m_idx is None:
            fact_store[v_idx] = output_tokens[0]

    # Rule extraction: directly from clean encoded outputs (no memory recall).
    for m_offset, modifier in enumerate(MODIFIERS):
        m_idx = n_verbs + m_offset
        matching = [(v, mi, out) for (v, mi, out) in train if mi == m_idx]

        patterns = []
        for v_idx, _, output_tokens in matching:
            encoded_out = _encode_output(reasoner, output_role_hvs, output_tokens)
            verb_out_sym = out_offset + v_idx
            patterns.append(unbind(encoded_out, reasoner.symbols[verb_out_sym]))
        pattern = torch.stack(patterns).mean(dim=0)

        residuals = []
        for v_idx, _, output_tokens in matching:
            encoded_out = _encode_output(reasoner, output_role_hvs, output_tokens)
            verb_out_sym = out_offset + v_idx
            verb_sym_hv = reasoner.symbols[verb_out_sym]
            residuals.append(encoded_out - bind(pattern, verb_sym_hv))
        residual = torch.stack(residuals).mean(dim=0)

        reasoner.modifier_patterns[modifier] = pattern
        reasoner.modifier_residuals[modifier] = residual

    out_cb = reasoner.symbols.all()[out_offset:]

    # train sample
    sample = train[:: max(1, len(train) // 20)]
    n_train_correct = 0
    for v_idx, m_idx, expected in sample:
        if m_idx is None:
            if [fact_store[v_idx]] == expected:
                n_train_correct += 1
        else:
            modifier = MODIFIERS[m_idx - n_verbs]
            verb_out_sym = fact_store[v_idx]
            out_hv = reasoner.apply_modifier_pattern(modifier, verb_out_sym)
            predicted = []
            for i in range(len(expected)):
                slot_hv = unbind(out_hv, output_role_hvs[i])
                pred_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
                predicted.append(pred_sym)
            if predicted == expected:
                n_train_correct += 1
    train_acc = n_train_correct / len(sample)

    # test: held-out compositions
    n_test_correct = 0
    for v_idx, m_idx, expected in test:
        modifier = MODIFIERS[m_idx - n_verbs]
        verb_out_sym = fact_store[v_idx]  # direct dict lookup, no recall noise
        out_hv = reasoner.apply_modifier_pattern(modifier, verb_out_sym)
        predicted = []
        for i in range(len(expected)):
            slot_hv = unbind(out_hv, output_role_hvs[i])
            pred_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
            predicted.append(pred_sym)
        if predicted == expected:
            n_test_correct += 1
    test_acc = n_test_correct / len(test) if test else 0.0

    return CapacityResultV3(
        n_verbs=n_verbs, d=d,
        train_size=len(train), test_size=len(test),
        train_acc=train_acc, test_acc=test_acc,
        elapsed_s=time.time() - t0,
    )


def sweep_v3(
    n_verbs_grid: Iterable[int] = (5, 20, 80, 320, 1280, 5000),
    d_grid: Iterable[int] = (2048, 4096, 8192, 16384),
    n_seeds: int = 3,
) -> list:
    rows = []
    for d in d_grid:
        row = []
        for nv in n_verbs_grid:
            seed_results = [_run_capacity_v3(nv, d, seed=s) for s in range(n_seeds)]
            mean_test = sum(r.test_acc for r in seed_results) / n_seeds
            mean_train = sum(r.train_acc for r in seed_results) / n_seeds
            mean_elapsed = sum(r.elapsed_s for r in seed_results) / n_seeds
            print(
                f"D={d:5d}  n_verbs={nv:5d}  train_acc={mean_train:.3f}  "
                f"test_acc={mean_test:.3f}  ({mean_elapsed:.1f}s/seed)",
                flush=True,
            )
            row.append(CapacityResultV3(
                n_verbs=nv, d=d,
                train_size=seed_results[0].train_size,
                test_size=seed_results[0].test_size,
                train_acc=mean_train, test_acc=mean_test,
                elapsed_s=mean_elapsed,
            ))
        rows.append(row)
    return rows


def format_table(rows) -> str:
    if not rows:
        return ""
    n_verbs_list = [c.n_verbs for c in rows[0]]
    out = ["| D \\ n_verbs | " + " | ".join(str(nv) for nv in n_verbs_list) + " |"]
    out.append("|---|" + "|".join(["---"] * len(n_verbs_list)) + "|")
    for row in rows:
        d = row[0].d
        accs = [f"{c.test_acc:.2f}" for c in row]
        out.append(f"| **D={d}** | " + " | ".join(accs) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    results = sweep_v3(n_seeds=args.seeds)
    print("\n## v3 (facts in dict, rules in VSA) test accuracy:")
    print(format_table(results))
