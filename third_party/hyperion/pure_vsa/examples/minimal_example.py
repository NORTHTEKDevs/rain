"""Minimal self-contained example of the HyperionReasoner public API.

Run:  python -m pure_vsa.examples.minimal_example
"""

from __future__ import annotations

from pure_vsa import Example, HyperionConfig, HyperionReasoner

# Toy vocabulary indices.
# Verbs:    walk=0, jump=1, swim=2  (input symbols)
# Modifier: twice=3                  (input symbol)
# Outputs:  W=4, J=5, S=6            (output symbols, contiguous block)

ROLE_VERB = 0
ROLE_MOD = 1

cfg = HyperionConfig(
    d=2048,                       # hypervector dimension
    n_symbols=7,                  # total symbol codebook size (input + output)
    n_input_roles=2,              # ROLE_VERB, ROLE_MOD
    max_output_len=3,             # we never produce sequences longer than 3
    output_vocab_offset=4,        # output tokens start at symbol index 4
)
reasoner = HyperionReasoner(cfg)

training = [
    # bare verbs
    Example({ROLE_VERB: 0}, [4]),                  # walk -> [W]
    Example({ROLE_VERB: 1}, [5]),                  # jump -> [J]
    Example({ROLE_VERB: 2}, [6]),                  # swim -> [S]   (BARE only -- modifier compositions held out)
    # (verb, twice) demonstrations -- but NOT for swim
    Example({ROLE_VERB: 0, ROLE_MOD: 3}, [4, 4]),  # walk twice -> [W, W]
    Example({ROLE_VERB: 1, ROLE_MOD: 3}, [5, 5]),  # jump twice -> [J, J]
]

rule_sizes = reasoner.fit(training, modifier_role_idx=ROLE_MOD, verb_role_idx=ROLE_VERB)
print(f"trained: {reasoner}")
print(f"extracted rules (modifier_symbol -> training examples used): {dict(rule_sizes)}")

# Held-out: swim twice. swim alone was in training; swim+twice was NOT.
predicted = reasoner.predict(
    {ROLE_VERB: 2, ROLE_MOD: 3},
    output_length=2,
    modifier_role_idx=ROLE_MOD,
    verb_role_idx=ROLE_VERB,
)
print(f"\npredicted swim+twice: {predicted}  (expected [6, 6] = [S, S])")
assert predicted == [6, 6], f"got {predicted}"
print("PASS -- compositional generalization to a held-out (verb, modifier) pair.")
