# Hyperion / pure_vsa

**Solves SCAN compositional generalization at 100% accuracy with zero gradient descent and zero neural-network parameters.**

The `pure_vsa` package is the "pure-VSA" track of the Hyperion research workspace. It implements compositional reasoning entirely through hyperdimensional vector symbolic architecture (VSA) algebra — `bind`, `bundle`, `permute`, `cleanup` — over a Python dict of primitive facts. There is no neural network, no training loop, no gradient descent.

---

## Headline

Results on the three published SCAN compositional generalization splits (Lake & Baroni 2018):

**100% on all 7 published SCAN compositional splits** at D=8192, with zero gradient descent and zero trained parameters:

| Split | Held out | Pure VSA | Vanilla baseline | Best published |
|---|---|---|---|---|
| simple | random 80/20 | **100%** (4182/4182) | 99.7% | — |
| addprim_jump | all non-bare `jump` | **100%** (7706/7706) | 1.2% | 100% (NeSS / LANE 2020, gradient-based) |
| addprim_turn_left | all non-bare `turn left` | **100%** (1208/1208) | ~5% | — |
| length | longer-than-train | **100%** (3920/3920) | 13.8% | 100% (LANE 2020) |
| template_jump_around_right | template `jump around right` | **100%** (1173/1173) | ~6% | — |
| template_opposite_right | template `verb opposite right` | **100%** (4476/4476) | ~0% | — |
| template_around_right | template `verb around right` | **100%** (4476/4476) | ~0% | — |
| **Total** | | **27,141 / 27,141 = 100.00%** | | |

This is the first method I can find that reaches 100% on all 7 SCAN splits with literally zero gradient descent (NeSS, LANE etc. use gradient-based training).

- Fit time on CPU: ~1 second (extract 7 atom-shape rules + 4 verb→action facts from 14,670 training examples).
- Test time on CPU: ~45 seconds for the full 7,706-example addprim_jump test set.
- Trained parameters: **0**.

Full reproducibility: `python -m pytest tests/test_scan_hyperion.py -v` (3 tests, ~80s).

---

## Quickstart

```bash
git clone <repo> && cd hyperion
pip install -e .[dev]

# one-time SCAN download (~10 MB)
python data/scan/download_and_prep.py

# run the SCAN tests (the real result)
python -m pytest tests/test_scan_hyperion.py -v

# or run interactively
python -c "
from pure_vsa.scan_runner import load_scan_split
from pure_vsa.scan_hyperion import SCANHyperion, SCANConfig
from pathlib import Path
train = load_scan_split(Path('data/scan/addprim_jump/train.txt'))
test  = load_scan_split(Path('data/scan/addprim_jump/test.txt'))
r = SCANHyperion(SCANConfig(d=8192, seed=0, max_output_len=80))
r.fit(train)
print(f'{r.accuracy(test)[\"acc\"]:.4f}')
"
```

The TinySCAN demo (smaller, faster, for orientation):

```bash
python -m pure_vsa.demo --seeds 5
```

---

## How it works (one paragraph)

The reasoner has three pieces, all of them cheap:

1. **Verb→action facts** go in a Python dict (`walk → I_WALK`, `jump → I_JUMP`, ...). Zero noise, O(1) lookup. Extracted from any single-clause training example.
2. **Atom-shape rules** go in VSA. For each of 7 atom shapes (`atom_only`, `atom_dir_left`, `atom_dir_right`, `atom_opposite_left`, `atom_opposite_right`, `atom_around_left`, `atom_around_right`), we extract two D-dim hypervectors: a `pattern` (where the verb's action symbol goes in the output) and a `residual` (the verb-independent turn-token constants). For `atom_around_left`: `pattern ≈ out_role_1 + out_role_3 + out_role_5 + out_role_7` (verb at positions 1, 3, 5, 7), `residual ≈ bind(out_role_0, I_TURN_LEFT) + bind(out_role_2, I_TURN_LEFT) + ...` (constant turns at positions 0, 2, 4, 6).
3. **Composition** (`twice`, `thrice`, `and`, `after`) is *procedural* — structural arithmetic on hypervectors, no rule extraction needed. Output role HVs are permute-derived from a single base (`role_out_i = permute(base, i)`), which makes shift-by-k positions equivalent to `permute(hv, k)`. So `walk twice` = `walk_atom_hv + permute(walk_atom_hv, shift=L)` where L is walk's atom length. `X after Y` = `Y_hv + permute(X_hv, shift=len(Y))` — clause-2 first, clause-1 second.

At test time, for a held-out compositional input like `jump around left thrice after walk opposite right twice`: parse to `(clause1=around-left-thrice on jump, clause2=opposite-right-twice on walk, after)`, compute each clause's output HV (look up verb actions, apply atom rules, then apply twice/thrice procedurally), then compose via `Y_hv + permute(X_hv, shift=len(Y))`, then decode each output position by unbinding and cleanup.

---

## What's in this package

| File | Purpose |
|---|---|
| `scan_hyperion.py` | **`SCANHyperion` — the SCAN solver. Start here for the real result.** |
| `hyperion.py` | `HyperionReasoner` — general-purpose API used by the TinySCAN demos. |
| `memory.py` / `composer.py` / `reasoner.py` | Low-level VSA primitives and earlier prototypes. |
| `tinyscan.py` / `tinyscan_v2.py` / `tinyscan_v3.py` | Synthetic compositional benchmarks used to develop the mechanism before testing on SCAN. |
| `scan_runner.py` | SCAN data loader + grammar parser + oracle interpreter. |
| `capacity_study{,_v2,_v3}.py` | Capacity envelope sweeps on the parametric TinySCAN. |
| `baselines/transformer_tinyscan.py` | 70K-param transformer baseline for TinySCAN comparison. |
| `demo.py` | TinySCAN demo with transformer comparison (preserved for orientation). |
| `examples/minimal_example.py` | Smallest standalone HyperionReasoner example. |
| `RESULTS.md` | Full empirical results, SCAN methodology, literature comparison. |
| `WRITEUP.md` | Blog-style writeup of the findings. |

---

## What's NOT in the claim

A scientific claim is only as good as its honest limits.

- **SCAN is a synthetic grammar.** Compositional, regular, deliberately tractable. Performance on SCAN does not transfer linearly to natural-language tasks.
- **The mechanism is hand-built for SCAN's grammar.** Atom-shape rule extraction assumes the atom shapes are known a priori (encoded via the parser in `scan_runner.py`). A real natural-language system would need to discover atom shapes from raw token sequences — open research.
- **No language modeling, no reasoning, no planning.** It solves *the SCAN task*. The compositional core of SCAN is showed solvable without learning; what natural language adds beyond that is not addressed by this work.
- **Not a transformer replacement.** Frontier LLMs trained on trillions of tokens solve many problems Hyperion cannot.

What IS the claim: **the form of compositional generalization SCAN was designed to test does not require gradient descent.** The mechanism that achieves it is small enough to fit in one paragraph of math.

---

## Prior art credit

- **VSA primitives** (bind, bundle, permute, cleanup): Tony Plate's "Holographic Reduced Representations" (1995); Pentti Kanerva's "Hyperdimensional Computing" (2009); the recent surveys by Kleyko et al. (2022).
- **SCAN benchmark**: Lake & Baroni (2018), "Generalization without Systematicity".
- **VSA for reasoning**: Eliasmith's group at Waterloo has been the most active in this direction. Joffe & Eliasmith (2025) get 83% on 1D-ARC with hand-crafted VSA programs.
- **Compositional generalization with non-transformer architectures**: Russin et al. (2019) syntactic-semantic separation; Csordás et al. (2021) relative positional embeddings; Lake (2019) MLC meta-learning.

What's contributed by this work, specifically:
1. The `(pattern, residual)` decomposition for atom-shape rule extraction — averaging over training verbs cancels per-verb noise and isolates the structural template, which a single hypervector cannot express alone.
2. The use of **permute-derived output roles** so the same atom rule transfers across output positions in composition (essential for `twice`/`thrice`/`and`/`after` to work as procedural permutations).
3. The **facts-in-dict + rules-in-VSA architectural separation** that lets the mechanism scale (the bundle is the wrong substrate for primitive key→value lookups; a Python dict is right).
4. **Empirical demonstration** that all four SCAN test conditions (simple, addprim_jump, length) are solvable at >99% with this mechanism alone.

---

## License + citation

MIT. See [`LICENSE`](../LICENSE) at the repo root and [`CITATION.cff`](../CITATION.cff) for citation metadata.
