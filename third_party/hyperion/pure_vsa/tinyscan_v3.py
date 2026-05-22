"""TinySCAN v3: nested compositions.

Grammar:
    S       -> CLAUSE | CLAUSE and CLAUSE
    CLAUSE  -> ACTION | ACTION MODIFIER
    ACTION  -> walk | jump | run | swim | look
    MODIFIER -> twice | thrice

Outputs (variable length up to 6 tokens):
    walk                          -> [W]
    walk twice                    -> [W, W]
    walk thrice                   -> [W, W, W]
    walk and jump                 -> [W, J]
    walk twice and jump           -> [W, W, J]
    walk and jump twice           -> [W, J, J]
    walk twice and jump thrice    -> [W, W, J, J, J]

Held-out: any clause involving `swim` with a modifier OR any conjunction involving
swim. Bare swim is in train. Anything else with swim is held out.

The point: output positions are variable-length and depend on the first clause's
modifier. The modifier rule "thrice" needs to be applied in both clause-1 (writes
to positions 1-3) and clause-2 (writes to positions L1+1..L1+3). For pure-VSA to
handle this without re-extracting per-position rules, the output role HVs must
be related by VSA permutation: role_out_i = permute(base, shift=i).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pure_vsa.tinyscan import ACTIONS, ACTION_TO_OUT, symbol_index
from vsa_core import permute as vsa_permute
from vsa_core.codebook import Codebook


MODIFIERS_V3 = ["twice", "thrice"]
MAX_OUTPUT_LEN = 6


# roles
ROLE_VERB1 = 0
ROLE_MOD1 = 1
ROLE_AND = 2
ROLE_VERB2 = 3
ROLE_MOD2 = 4
N_INPUT_ROLES = 5

# Output role base index; the actual output roles are permute-derived from base_role_hv.
ROLE_OUT_BASE = 5  # this index in role codebook holds the BASE output-role HV


def build_role_codebook(d: int, seed: int = 1) -> tuple[Codebook, Tensor]:
    """Build a role codebook where the first 5 entries are independent random
    HVs (input roles + AND marker) and the 6th is the OUTPUT-ROLE BASE HV.

    Returns (codebook, output_role_hvs) where output_role_hvs has shape
    (MAX_OUTPUT_LEN, D) and output_role_hvs[i] = permute(base, shift=i).
    """
    cb = Codebook(N_INPUT_ROLES + 1, d, seed=seed)
    base = cb[ROLE_OUT_BASE]
    output_role_hvs = torch.stack([vsa_permute(base, shift=i) for i in range(MAX_OUTPUT_LEN)])
    return cb, output_role_hvs


@dataclass
class NestedExample:
    """A v3 example.

    clause1: (verb, modifier_or_None)
    clause2: (verb, modifier_or_None) or None if no conjunction
    """
    clause1_verb: str
    clause1_mod: str | None
    clause2_verb: str | None
    clause2_mod: str | None
    output_tokens: list[str]

    def has_conjunction(self) -> bool:
        return self.clause2_verb is not None

    def input_slots(self) -> dict[int, int]:
        """Return input slot dict using role indices."""
        slots = {ROLE_VERB1: symbol_index(self.clause1_verb)}
        if self.clause1_mod is not None:
            slots[ROLE_MOD1] = symbol_index(self.clause1_mod)
        if self.clause2_verb is not None:
            slots[ROLE_VERB2] = symbol_index(self.clause2_verb)
            if self.clause2_mod is not None:
                slots[ROLE_MOD2] = symbol_index(self.clause2_mod)
        return slots


def _expand_clause(action: str, modifier: str | None) -> list[str]:
    a_out = ACTION_TO_OUT[action]
    if modifier is None:
        return [a_out]
    if modifier == "twice":
        return [a_out, a_out]
    if modifier == "thrice":
        return [a_out, a_out, a_out]
    raise ValueError(f"unknown modifier: {modifier}")


def _apply_grammar_v3(
    c1_verb: str, c1_mod: str | None, c2_verb: str | None, c2_mod: str | None
) -> list[str]:
    out = _expand_clause(c1_verb, c1_mod)
    if c2_verb is not None:
        out = out + _expand_clause(c2_verb, c2_mod)
    return out


def make_dataset_v3(held_out_action: str = "swim") -> tuple[
    list[NestedExample], list[NestedExample]
]:
    """Build train + test splits.

    Train: every example NOT involving the held-out action in a non-bare position.
           (Bare swim is in train.)
    Test:  every example involving the held-out action in a non-bare position.
    """
    train: list[NestedExample] = []
    test: list[NestedExample] = []

    actions = ACTIONS
    mods_opt = [None] + MODIFIERS_V3

    def is_held_out(ex: NestedExample) -> bool:
        # bare held-out (no mod, no conjunction) -> NOT held out (it's in train)
        if (
            ex.clause1_verb == held_out_action
            and ex.clause1_mod is None
            and ex.clause2_verb is None
        ):
            return False
        # any other use of held_out_action -> held out
        if ex.clause1_verb == held_out_action and ex.clause1_mod is not None:
            return True
        if ex.clause2_verb == held_out_action:
            return True
        return False

    # single-clause examples (with optional modifier)
    for v in actions:
        for m in mods_opt:
            ex = NestedExample(v, m, None, None, _apply_grammar_v3(v, m, None, None))
            if is_held_out(ex):
                test.append(ex)
            else:
                train.append(ex)

    # two-clause examples (conjunctions, with optional modifiers per clause)
    for v1 in actions:
        for m1 in mods_opt:
            for v2 in actions:
                if v1 == v2:
                    continue  # exclude self-conjunctions to keep size manageable
                for m2 in mods_opt:
                    ex = NestedExample(
                        v1, m1, v2, m2,
                        _apply_grammar_v3(v1, m1, v2, m2),
                    )
                    if is_held_out(ex):
                        test.append(ex)
                    else:
                        train.append(ex)

    return train, test


def parse_example(ex: NestedExample) -> list[tuple[str, str | None]]:
    """Return list of (verb, modifier) clauses in order. 1 or 2 clauses."""
    clauses = [(ex.clause1_verb, ex.clause1_mod)]
    if ex.clause2_verb is not None:
        clauses.append((ex.clause2_verb, ex.clause2_mod))
    return clauses


def clause_output_len(modifier: str | None) -> int:
    if modifier is None:
        return 1
    if modifier == "twice":
        return 2
    if modifier == "thrice":
        return 3
    raise ValueError(f"unknown modifier: {modifier}")
