"""Empirical capacity study: where does the pattern+residual mechanism break?

Sweeps:
  - n_verbs in {5, 10, 20, 40, 80}
  - D in {1024, 2048, 4096, 8192, 16384}
  - n_modifiers fixed at 4 (twice, thrice, left, right)

For each (n_verbs, D) combination:
  - Build the dataset.
  - Hold out one verb across all modifiers.
  - Train pure-VSA: store primitives, extract modifier rules.
  - Eval on held-out compositions.
  - Record accuracy.

Output: a table of n_verbs x D showing held-out accuracy.

This maps the operating regime of pure-VSA TinySCAN-style compositional
generalization.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from pure_vsa.reasoner import PureVSAReasoner
from vsa_core import bind, unbind
from vsa_core.cleanup import similarity
from vsa_core.codebook import Codebook


# 80 unique action stems for the sweep.
MAX_ACTION_STEMS = [
    "walk", "jump", "run", "swim", "look", "climb", "slide", "dance",
    "sing", "sleep", "drink", "eat", "talk", "write", "draw", "paint",
    "throw", "catch", "kick", "push", "pull", "lift", "carry", "drop",
    "open", "close", "lock", "unlock", "break", "fix", "build", "destroy",
    "drive", "ride", "fly", "sail", "row", "skate", "ski", "surf",
    "hike", "crawl", "stretch", "bend", "twist", "turn", "spin", "roll",
    "wave", "clap", "snap", "tap", "knock", "stomp", "shake", "nod",
    "smile", "frown", "laugh", "cry", "yell", "whisper", "shout", "hum",
    "type", "click", "scroll", "swipe", "tap2", "press", "release", "hold",
    "fold", "unfold", "tear", "stitch", "sew", "knit", "weave", "cut",
]
assert len(MAX_ACTION_STEMS) >= 80

MODIFIERS = ["twice", "thrice", "left", "right"]


# role indices for the parameterized grammar
ROLE_VERB = 0
ROLE_MOD = 1
N_ROLES = 2
N_MAX_OUT = 4
# output role HVs are permute-derived for nested-rule reuse (kept here for generality
# though this study uses single-clause only).


@dataclass
class CapacityResult:
    n_verbs: int
    d: int
    train_size: int
    test_size: int
    train_acc: float
    test_acc: float
    elapsed_s: float


def _build_dataset(n_verbs: int, held_out_verb_idx: int) -> tuple[list, list]:
    """Return (train, test) examples.

    Each example is a tuple (verb_idx, modifier_idx_or_None, output_token_indices).
    Output tokens are: for bare verb -> [verb_out_idx]; twice -> [v,v]; thrice -> [v,v,v];
                       left -> [LEFT, v]; right -> [RIGHT, v].
    """
    train = []
    test = []
    n_modifiers = len(MODIFIERS)

    # symbol layout: 0..n_verbs-1 = input verbs; n_verbs..n_verbs+n_modifiers-1 = modifiers;
    # n_verbs+n_modifiers..n_verbs+n_modifiers+n_verbs-1 = output action tokens (one per verb);
    # last 2 indices = LEFT, RIGHT.
    out_offset = n_verbs + n_modifiers
    LEFT_idx = out_offset + n_verbs
    RIGHT_idx = LEFT_idx + 1
    total_symbols = RIGHT_idx + 1

    def out_for_verb(v_idx: int) -> int:
        return out_offset + v_idx

    for v in range(n_verbs):
        # bare verb
        bare = (v, None, [out_for_verb(v)])
        if v == held_out_verb_idx:
            train.append(bare)  # bare held-out is in train
        else:
            train.append(bare)
        for m_offset, modifier in enumerate(MODIFIERS):
            m_idx = n_verbs + m_offset
            v_out = out_for_verb(v)
            if modifier == "twice":
                output = [v_out, v_out]
            elif modifier == "thrice":
                output = [v_out, v_out, v_out]
            elif modifier == "left":
                output = [LEFT_idx, v_out]
            elif modifier == "right":
                output = [RIGHT_idx, v_out]
            ex = (v, m_idx, output)
            if v == held_out_verb_idx:
                test.append(ex)
            else:
                train.append(ex)
    return train, test, total_symbols, LEFT_idx, RIGHT_idx, out_offset


def _encode_input_pair(reasoner: PureVSAReasoner, v_idx: int, m_idx: int | None) -> torch.Tensor:
    """Build the input key for a (verb, modifier) example."""
    terms = [bind(reasoner.roles[ROLE_VERB], reasoner.symbols[v_idx])]
    if m_idx is not None:
        terms.append(bind(reasoner.roles[ROLE_MOD], reasoner.symbols[m_idx]))
    return torch.stack(terms).sum(dim=0)


def _encode_output(reasoner: PureVSAReasoner, output_role_hvs: torch.Tensor, tokens: list[int]) -> torch.Tensor:
    """Encode an output token list as bundle of bind(role_i, token_sym)."""
    return torch.stack([bind(output_role_hvs[i], reasoner.symbols[t]) for i, t in enumerate(tokens)]).sum(dim=0)


def _run_capacity_one(n_verbs: int, d: int, seed: int = 0) -> CapacityResult:
    """Run a single (n_verbs, D) capacity experiment."""
    torch.manual_seed(seed)
    t0 = time.time()

    held_out_verb_idx = 0  # always hold out the first verb
    train, test, total_symbols, LEFT_idx, RIGHT_idx, out_offset = _build_dataset(
        n_verbs, held_out_verb_idx
    )

    symbols = Codebook(total_symbols, d, seed=seed)
    roles = Codebook(N_ROLES, d, seed=seed + 1)
    reasoner = PureVSAReasoner(d=d, symbol_codebook=symbols, role_codebook=roles)

    # output role HVs (4 positions max -- enough for thrice + LEFT/RIGHT)
    from vsa_core import permute as vsa_permute  # noqa: PLC0415
    base = roles[ROLE_VERB]  # use ROLE_VERB as base; arbitrary choice
    output_role_hvs = torch.stack([vsa_permute(base, shift=i) for i in range(N_MAX_OUT)])

    # store all training examples
    for v_idx, m_idx, output_tokens in train:
        key = _encode_input_pair(reasoner, v_idx, m_idx)
        value = _encode_output(reasoner, output_role_hvs, output_tokens)
        reasoner.memory.store(torch.sign(key), value)

    # extract modifier rules
    for m_offset, modifier in enumerate(MODIFIERS):
        m_idx = n_verbs + m_offset
        # find training examples for this modifier
        matching = [(v, mi, out) for (v, mi, out) in train if mi == m_idx]
        examples_for_rule = []
        for v_idx, _, _ in matching:
            input_slots = {ROLE_VERB: v_idx, ROLE_MOD: m_idx}
            verb_out_sym = out_offset + v_idx
            examples_for_rule.append((input_slots, verb_out_sym))
        reasoner.extract_modifier_pattern(modifier, examples_for_rule)

    # eval train (memorization sanity check on a sample)
    sample = train[:: max(1, len(train) // 20)]
    n_train_correct = 0
    for v_idx, m_idx, expected in sample:
        if m_idx is None:
            key_hard = torch.sign(_encode_input_pair(reasoner, v_idx, None))
            raw = reasoner.memory.retrieve_raw(key_hard)
            slot_hv = unbind(raw, output_role_hvs[0])
            out_cb = reasoner.symbols.all()[out_offset:]
            pred = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
            if [pred] == expected:
                n_train_correct += 1
        else:
            # use modifier rule to predict
            modifier = MODIFIERS[m_idx - n_verbs]
            verb_out_sym = out_offset + v_idx
            out_hv = reasoner.apply_modifier_pattern(modifier, verb_out_sym)
            predicted = []
            for i in range(len(expected)):
                slot_hv = unbind(out_hv, output_role_hvs[i])
                out_cb = reasoner.symbols.all()[out_offset:]
                pred_sym = out_offset + similarity(slot_hv, out_cb).argmax(dim=-1).item()
                predicted.append(pred_sym)
            if predicted == expected:
                n_train_correct += 1
    train_acc = n_train_correct / len(sample)

    # eval test (compositional generalization on held-out verb)
    n_test_correct = 0
    for v_idx, m_idx, expected in test:
        modifier = MODIFIERS[m_idx - n_verbs]
        # look up held-out verb's output symbol from bare recall
        bare_key = torch.sign(_encode_input_pair(reasoner, v_idx, None))
        bare_recall = reasoner.memory.retrieve_raw(bare_key)
        slot_hv = unbind(bare_recall, output_role_hvs[0])
        out_cb = reasoner.symbols.all()[out_offset:]
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

    elapsed = time.time() - t0
    return CapacityResult(
        n_verbs=n_verbs,
        d=d,
        train_size=len(train),
        test_size=len(test),
        train_acc=train_acc,
        test_acc=test_acc,
        elapsed_s=elapsed,
    )


def sweep(
    n_verbs_grid: Iterable[int] = (5, 10, 20, 40, 80),
    d_grid: Iterable[int] = (1024, 2048, 4096, 8192, 16384),
    n_seeds: int = 3,
) -> list[list[CapacityResult]]:
    """Run the full sweep. Returns list[d][n_verbs] of mean results."""
    rows = []
    for d in d_grid:
        row = []
        for nv in n_verbs_grid:
            seed_results = [_run_capacity_one(nv, d, seed=s) for s in range(n_seeds)]
            mean_test = sum(r.test_acc for r in seed_results) / n_seeds
            mean_train = sum(r.train_acc for r in seed_results) / n_seeds
            mean_elapsed = sum(r.elapsed_s for r in seed_results) / n_seeds
            print(
                f"D={d:5d}  n_verbs={nv:3d}  train_acc={mean_train:.3f}  "
                f"test_acc={mean_test:.3f}  ({mean_elapsed:.1f}s/seed, n={n_seeds})"
            )
            row.append(CapacityResult(
                n_verbs=nv, d=d,
                train_size=seed_results[0].train_size,
                test_size=seed_results[0].test_size,
                train_acc=mean_train,
                test_acc=mean_test,
                elapsed_s=mean_elapsed,
            ))
        rows.append(row)
    return rows


def format_table(rows: list[list[CapacityResult]]) -> str:
    """Pretty-print the test_acc matrix as a markdown table."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller sweep")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    if args.quick:
        results = sweep(
            n_verbs_grid=(5, 10, 20, 40),
            d_grid=(1024, 2048, 4096, 8192),
            n_seeds=args.seeds,
        )
    else:
        results = sweep(n_seeds=args.seeds)
    print("\n## Test accuracy matrix (held-out verb's compositional generalization):")
    print(format_table(results))
