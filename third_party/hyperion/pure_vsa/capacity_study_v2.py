"""Capacity study v2: extract rules from CLEAN encoded outputs, not from memory recalls.

Hypothesis: the bottleneck in capacity_study v1 is rule-extraction noise (the
modifier rule is extracted via `unbind(memory_recall(V, M), V_sym)` and memory
recall carries cross-talk from all other stored items). If we extract the rule
directly from the encoded output (which is clean at training time, no recall
needed), the noise stage is removed and capacity should scale further.

This is architecturally honest: the system observes (input, output) pairs at
training time and can extract rules directly from observations. Memory is then
used only for bare-verb lookup at test time.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch

# reuse the v1 dataset builder via import
from pure_vsa.capacity_study import (
    MODIFIERS,
    N_MAX_OUT,
    N_ROLES,
    ROLE_VERB,
    _build_dataset,
    _encode_input_pair,
    _encode_output,
)
from pure_vsa.reasoner import PureVSAReasoner
from vsa_core import bind, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


@dataclass
class CapacityResultV2:
    n_verbs: int
    d: int
    train_size: int
    test_size: int
    train_acc: float
    test_acc: float
    elapsed_s: float


def _run_capacity_v2(n_verbs: int, d: int, seed: int = 0) -> CapacityResultV2:
    """Same setup as v1, but rule extraction uses clean encoded outputs."""
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

    # Store ONLY bare verb examples in memory (used for test-time lookup of
    # held-out verb's output symbol). All modifier examples are NOT stored;
    # we extract their rules directly from encoded outputs below.
    for v_idx, m_idx, output_tokens in train:
        if m_idx is not None:
            continue  # skip modifier examples in memory
        key = _encode_input_pair(reasoner, v_idx, m_idx)
        value = _encode_output(reasoner, output_role_hvs, output_tokens)
        reasoner.memory.store(torch.sign(key), value)

    # Extract modifier rules DIRECTLY from clean encoded outputs (no memory).
    for m_offset, modifier in enumerate(MODIFIERS):
        m_idx = n_verbs + m_offset
        matching = [(v, mi, out) for (v, mi, out) in train if mi == m_idx]

        patterns = []
        residuals = []
        for v_idx, _, output_tokens in matching:
            encoded_out = _encode_output(reasoner, output_role_hvs, output_tokens)
            verb_out_sym = out_offset + v_idx
            verb_sym_hv = reasoner.symbols[verb_out_sym]
            patterns.append(unbind(encoded_out, verb_sym_hv))
        pattern = torch.stack(patterns).mean(dim=0)

        for v_idx, _, output_tokens in matching:
            encoded_out = _encode_output(reasoner, output_role_hvs, output_tokens)
            verb_out_sym = out_offset + v_idx
            verb_sym_hv = reasoner.symbols[verb_out_sym]
            predicted = bind(pattern, verb_sym_hv)
            residuals.append(encoded_out - predicted)
        residual = torch.stack(residuals).mean(dim=0)

        reasoner.modifier_patterns[modifier] = pattern
        reasoner.modifier_residuals[modifier] = residual

    # Train acc: sample some training examples, predict, compare.
    sample = train[:: max(1, len(train) // 20)]
    n_train_correct = 0
    out_cb = reasoner.symbols.all()[out_offset:]
    for v_idx, m_idx, expected in sample:
        if m_idx is None:
            key_hard = torch.sign(_encode_input_pair(reasoner, v_idx, None))
            raw = reasoner.memory.retrieve_raw(key_hard)
            slot_hv = unbind(raw, output_role_hvs[0])
            pred = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
            if [pred] == expected:
                n_train_correct += 1
        else:
            modifier = MODIFIERS[m_idx - n_verbs]
            verb_out_sym = out_offset + v_idx
            out_hv = reasoner.apply_modifier_pattern(modifier, verb_out_sym)
            predicted = []
            for i in range(len(expected)):
                slot_hv = unbind(out_hv, output_role_hvs[i])
                pred_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
                predicted.append(pred_sym)
            if predicted == expected:
                n_train_correct += 1
    train_acc = n_train_correct / len(sample)

    # Test acc: held-out compositions
    n_test_correct = 0
    for v_idx, m_idx, expected in test:
        modifier = MODIFIERS[m_idx - n_verbs]
        bare_key = torch.sign(_encode_input_pair(reasoner, v_idx, None))
        bare_recall = reasoner.memory.retrieve_raw(bare_key)
        slot_hv = unbind(bare_recall, output_role_hvs[0])
        verb_out_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()

        out_hv = reasoner.apply_modifier_pattern(modifier, verb_out_sym)
        predicted = []
        for i in range(len(expected)):
            slot_hv = unbind(out_hv, output_role_hvs[i])
            pred_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
            predicted.append(pred_sym)
        if predicted == expected:
            n_test_correct += 1
    test_acc = n_test_correct / len(test) if test else 0.0

    return CapacityResultV2(
        n_verbs=n_verbs, d=d,
        train_size=len(train), test_size=len(test),
        train_acc=train_acc, test_acc=test_acc,
        elapsed_s=time.time() - t0,
    )


def sweep_v2(
    n_verbs_grid: Iterable[int] = (5, 10, 20, 40, 80, 160),
    d_grid: Iterable[int] = (1024, 2048, 4096, 8192, 16384),
    n_seeds: int = 3,
) -> list:
    rows = []
    for d in d_grid:
        row = []
        for nv in n_verbs_grid:
            seed_results = [_run_capacity_v2(nv, d, seed=s) for s in range(n_seeds)]
            mean_test = sum(r.test_acc for r in seed_results) / n_seeds
            mean_train = sum(r.train_acc for r in seed_results) / n_seeds
            mean_elapsed = sum(r.elapsed_s for r in seed_results) / n_seeds
            print(
                f"D={d:5d}  n_verbs={nv:3d}  train_acc={mean_train:.3f}  "
                f"test_acc={mean_test:.3f}  ({mean_elapsed:.1f}s/seed)"
            )
            row.append(CapacityResultV2(
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
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    results = sweep_v2(n_seeds=args.seeds)
    print("\n## v2 (clean-extraction) test accuracy:")
    print(format_table(results))
