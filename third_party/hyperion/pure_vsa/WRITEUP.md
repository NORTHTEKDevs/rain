# Solve SCAN without learning

*A pure algebraic compositional reasoner that achieves 100% on the canonical SCAN compositional generalization benchmark — where vanilla transformers get 1.2%.*

---

## The thing

**Where this fits in the literature.** Two prior methods have achieved 100% on SCAN add-primitive jump: NeSS (Chen et al. NeurIPS 2020), a neural-symbolic stack machine trained with curriculum + neural controller; and LANE (Liu et al. NeurIPS 2020), a Composer + Solver architecture trained with hierarchical reinforcement learning. Both use gradient-based training. The method described here is the first I can find that reaches 100% on the same task with **literally zero gradient descent** — no neural network, no optimizer, no learning loop. The training pass is a one-shot extraction of facts and hypervector rule patterns in ~1 second on a CPU.

[SCAN](https://github.com/brendenlake/SCAN) is the benchmark Lake and Baroni introduced in 2018 to expose the compositional generalization failure of neural sequence models. The grammar is small and regular:

```
S       -> CLAUSE | CLAUSE and CLAUSE | CLAUSE after CLAUSE
CLAUSE  -> ATOM | ATOM twice | ATOM thrice
ATOM    -> VERB | VERB DIR | VERB opposite DIR | VERB around DIR
VERB    -> walk | look | run | jump | turn
DIR     -> left | right
```

Outputs are deterministic action sequences. `walk around left thrice and jump opposite right twice` produces a specific 30-token output that's fully determined by the grammar.

The hardest of the published splits, `addprim_jump`, holds out **every example involving `jump` except the single bare `jump → I_JUMP`**. The test set is 7,706 compositional examples like `jump around left twice and walk thrice`. The training set has 14,670 examples — none of which contain `jump` in any composition.

Lake and Baroni's vanilla seq2seq baseline gets **1.2%** on this split. Attention seq2seq: **0.08%**. Csordás et al.'s relative-position transformer (2021) gets ~78% on the best seeds. Lake's MLC meta-learning approach (2019) reaches >99% — but uses an active meta-training intervention.

The system in this repository solves it with **0%** of those things. There is no neural network, no gradient descent, no learning algorithm. Just three pieces of math:

```
SCAN addprim_jump (D=8192, seed=0):  100.00%  (7706/7706)
SCAN length:                          99.90%  (3916/3920)
SCAN simple:                         100.00%  (4182/4182)
```

Fit time: 1 second on a CPU. Test time: 45 seconds. Trained parameters: zero.

## How it works

Three building blocks.

**One: verb→action facts go in a Python dict.** Bare-verb examples in training give us `walk → I_WALK`, `look → I_LOOK`, `run → I_RUN`, `jump → I_JUMP`. The fact `jump → I_JUMP` is the *only* thing the system ever sees about `jump`. Everything else about jump-in-composition is predicted, not memorized.

**Two: atom-shape rules go in pure VSA.** SCAN has 7 atom shapes:

```
atom_only             :  verb              -> I_VERB
atom_dir_left/right   :  verb left/right   -> I_TURN_LEFT/RIGHT  I_VERB
atom_opposite_left/r  :  verb opposite L/R -> I_TURN_L/R  I_TURN_L/R  I_VERB
atom_around_left/r    :  verb around L/R   -> (I_TURN_L/R  I_VERB) x 4
```

For each shape, we extract two D-dim hypervectors from training examples (none of which contain `jump`):

```
pattern_shape  = mean over training verbs V of [ unbind(encoded_atom_output(V), V_action_HV) ]
residual_shape = mean over V of [ encoded_atom_output(V) - bind(pattern_shape, V_action_HV) ]
```

The `pattern` captures *where in the output the verb's action symbol goes*. The `residual` captures *the verb-independent turn-token constants*. For `atom_around_left`:

```
pattern  ~= out_role_1 + out_role_3 + out_role_5 + out_role_7         (verb at positions 1, 3, 5, 7)
residual ~= bind(out_role_0, I_TURN_LEFT) + bind(out_role_2, I_TURN_LEFT)
         + bind(out_role_4, I_TURN_LEFT) + bind(out_role_6, I_TURN_LEFT)  (turns at 0, 2, 4, 6)
```

At test time on `jump around left`: look up `jump → I_JUMP`, compute `bind(pattern_around_left, I_JUMP_HV) + residual_around_left`, decode each output position. The verb-independent residual stays. The verb-dependent pattern gets bound with the new verb's action. The 8-token output reads `I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP`.

**Three: composition is procedural VSA.** No rule extraction. Output role HVs are *permute-derived from a single base*: `role_out_i = permute(base, shift=i)`. With that choice, permutation distributes over binding:

```
permute(bind(role_i, X), k) = bind(role_{i+k}, X)
```

So shifting a sub-output by k positions in the output is identical to permuting its hypervector by k. The four composition operators become one-liners:

```
twice(X):        X_hv + permute(X_hv, shift=len(X))
thrice(X):       X_hv + permute(X_hv, len(X)) + permute(X_hv, 2*len(X))
X and Y:         X_hv + permute(Y_hv, shift=len(X))
X after Y:       Y_hv + permute(X_hv, shift=len(Y))           # clause-2 first
```

For `jump around left thrice after walk opposite right twice`:
1. Compute `jump_around_left_hv` (atom rule applied to jump).
2. Apply thrice: `jump_around_left_thrice_hv = jump_around_left_hv + permute(., 8) + permute(., 16)`.
3. Compute `walk_opposite_right_twice_hv` similarly.
4. Compose with after: `walk_opposite_right_twice_hv + permute(jump_around_left_thrice_hv, shift=6)`.
5. Decode each of the 30 output positions by unbinding the corresponding role and finding the nearest output token in the codebook.

Output matches the SCAN ground truth.

## Why it works

The thing that makes this work is **choosing the right primitives**. The pieces are:

- Random ±1 hypervectors of dimension 10⁴ — approximately orthogonal by construction.
- FFT-based circular convolution as `bind` — invertible, distributive over addition.
- Sum + sign as `bundle`, cyclic permutation as `permute` — both compose cleanly with bind.
- A Python dict for lookups that don't need structural querying — bypasses the bundle's cross-talk noise.

None of these are new. Tony Plate's 1995 PhD thesis described all the VSA primitives. Kanerva's 2009 paper named the field. Eliasmith's group has been showing for years that compositional reasoning is expressible in VSA.

What's new here is the empirical demonstration: **all three SCAN splits are solvable at literature-best accuracy** with the smallest possible mechanism on top of the standard primitives. The `(pattern, residual)` decomposition averages out per-verb noise to isolate structural rules. The permute-derived output roles make composition mechanical. Facts-in-dict keeps memory cross-talk out of the loop for primitive lookups.

I built this expecting it would fail on real SCAN — my first pass handled only 2 of 7,706 test cases. Then I extended the mechanism to cover SCAN's actual atom shapes, expecting partial success. The result was 99.58% on the first run, 100% with a higher hypervector dimension or different seed. The honest reaction was disbelief, then verifying there was no data leakage, then running on the other two splits (length: 99.90%; simple: 100%), then doing a literature comparison.

The mechanism is small. The result is large.

## What this is — and is not

**Is:**

- A concrete, runnable, deterministic mechanism that solves SCAN at literature-best accuracy with no neural network and **no gradient descent at all** (the prior 100% methods, NeSS and LANE, both use gradient-based training).
- A clean separation of facts (Python dict, O(1)) from rules (VSA hypervectors, O(D) per primitive). Composition is procedural arithmetic.
- A library small enough to read in an hour: ~1500 lines of Python, no dependencies beyond PyTorch and NumPy.
- An empirical existence proof that *the compositional generalization SCAN was designed to test is not a learning problem*. It's an algebraic substrate problem.

**Is not:**

- A general-purpose language model. The mechanism handles SCAN's grammar via a hand-written parser. Raw natural-language strings (or even COGS-style English-like sentences) need their own parsing stage.
- A "learning from scratch" system. The SCAN grammar — what verbs, modifiers, atom shapes, and composition operators exist — is encoded in the parser, not discovered. NeSS and LANE learn this part; pure VSA does not. What pure VSA does learn from training data: the verb→action token mappings and the atom-shape rule patterns. The compositional generalization (held-out `jump` compositions) is then pure algebra.
- A discovery of new VSA primitives. The algebra is from 1995–2009 (Plate, Kanerva).
- A replacement for transformers on open-ended tasks. Trillion-parameter LLMs trained on trillions of tokens do many things this system cannot.
- A test of multi-step reasoning, planning, or hierarchical abstraction. Those are the open research questions.
- Magic. The grammar provides the search space; the algebra navigates it without gradient descent.

**Open question:** does the same mechanism extend to natural-language tasks if a separate component does the parsing? A frontier LLM as parser + pure VSA as compositional reasoner could be the right hybrid. That's the next experiment.

## Reproducing

```bash
git clone <repo> && cd hyperion
pip install -e .[dev]
python data/scan/download_and_prep.py

# the headline result — 3 SCAN integration tests, ~80s on CPU
python -m pytest tests/test_scan_hyperion.py -v

# single-split eval
python -c "
from pure_vsa.scan_runner import load_scan_split
from pure_vsa.scan_hyperion import SCANHyperion, SCANConfig
from pathlib import Path
train = load_scan_split(Path('data/scan/addprim_jump/train.txt'))
test  = load_scan_split(Path('data/scan/addprim_jump/test.txt'))
r = SCANHyperion(SCANConfig(d=8192, seed=0, max_output_len=80))
r.fit(train)
print(f'{r.accuracy(test)[\"acc\"]:.4f}')
# -> 1.0000
"
```

Full empirical numbers across seeds and splits in [`RESULTS.md`](RESULTS.md).

Code: [github.com/NORTHTEKDevs/hyperion](https://github.com/NORTHTEKDevs/hyperion), `pure_vsa/` subdirectory. License: MIT.

If you find where this approach extends — or where it breaks in an instructive way — open an issue.
