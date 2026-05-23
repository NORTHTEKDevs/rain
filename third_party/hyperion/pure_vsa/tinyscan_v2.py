"""TinySCAN v2: adds binary "and" conjunction.

Grammar:
    S         -> CLAUSE | CLAUSE and CLAUSE
    CLAUSE    -> ACTION
    ACTION    -> walk | jump | run | swim | look

Outputs (token sequences):
    walk             -> [W]
    walk and jump    -> [W, J]
    run and swim     -> [R, S]
    swim and walk    -> [S, W]

Held-out split: any sentence involving swim *as part of a conjunction*.
Bare swim is in train. Swim+conjunction is not.

This tests whether the pattern+residual mechanism generalizes from 1-ary
(modifier) rules to 2-ary (conjunction) rules without architectural change.
"""

from __future__ import annotations

from dataclasses import dataclass

# reuse v1 symbols
from pure_vsa.tinyscan import (
    ACTION_TO_OUT,
    ACTIONS,
    symbol_index,
)

# additional role for the second verb in a conjunction
ROLE_VERB1 = 0  # alias of ROLE_VERB for clarity in v2
ROLE_VERB2 = 1  # was ROLE_MOD in v1, now repurposed for the second verb
ROLE_AND = 2
ROLE_OUT1 = 3
ROLE_OUT2 = 4
N_ROLES_V2 = 5

OUTPUT_ROLES_V2 = [ROLE_OUT1, ROLE_OUT2]


@dataclass
class ConjExample:
    verb1: str
    verb2: str | None  # None means bare (no conjunction)
    output_tokens: list[str]

    def input_slots(self) -> dict[int, int]:
        slots = {ROLE_VERB1: symbol_index(self.verb1)}
        if self.verb2 is not None:
            slots[ROLE_VERB2] = symbol_index(self.verb2)
        return slots

    def output_slots(self) -> dict[int, int]:
        return {
            OUTPUT_ROLES_V2[i]: symbol_index(tok)
            for i, tok in enumerate(self.output_tokens)
        }


def _apply_grammar_v2(verb1: str, verb2: str | None) -> list[str]:
    a_out1 = ACTION_TO_OUT[verb1]
    if verb2 is None:
        return [a_out1]
    a_out2 = ACTION_TO_OUT[verb2]
    return [a_out1, a_out2]


def make_dataset_v2(held_out_action: str = "swim") -> tuple[
    list[ConjExample], list[ConjExample]
]:
    """train = all bare actions + all (a, b) conjunctions where NEITHER a nor b is held out.
       test  = all (a, b) conjunctions where a == held_out or b == held_out (excluding bare).
    """
    train, test = [], []
    for action in ACTIONS:
        train.append(ConjExample(action, None, _apply_grammar_v2(action, None)))
    for a in ACTIONS:
        for b in ACTIONS:
            if a == b:
                continue  # exclude self-conjunctions to keep set small
            ex = ConjExample(a, b, _apply_grammar_v2(a, b))
            if a == held_out_action or b == held_out_action:
                test.append(ex)
            else:
                train.append(ex)
    return train, test
