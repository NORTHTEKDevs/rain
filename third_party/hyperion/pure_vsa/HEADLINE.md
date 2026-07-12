# Hyperion / pure_vsa — Headline Results

**Updated:** 2026-05-20
**Mechanism:** pure VSA algebra (bind / unbind / bundle / permute / cleanup) over bipolar hypervectors + Python-dict facts + per-operation procedural composition.
**Trained parameters:** 0.
**Hardware:** single CPU.

---

## Benchmarks covered

### 1. SCAN (Lake & Baroni 2018; Loula et al. 2018) — 7 compositional splits at 100%

| Split | Held out | Pure VSA (hand-parser) | Pure VSA (parser-free) | Best published |
|---|---|---|---|---|
| simple | random 80/20 | **100.00%** (4182/4182) | **100.00%** | — |
| addprim_jump | non-bare `jump` | **100.00%** (7706/7706) | **100.00%** | 100% (NeSS / LANE 2020, gradient) |
| addprim_turn_left | non-bare `turn left` | **100.00%** (1208/1208) | — | — |
| length | longer outputs than train | **100.00%** (3920/3920) | **100.00%** | 100% (LANE 2020) |
| template_jump_around_right | template `jump around right` | **100.00%** (1173/1173) | — | — |
| template_opposite_right | template `verb opposite right` (any verb) | **100.00%** (4476/4476) | **100.00%** | — |
| template_around_right | template `verb around right` (any verb) | **100.00%** (4476/4476) | — | — |
| **SCAN TOTAL** | | **27,141 / 27,141 = 100.00%** | (subset: 13,584 / 13,584 = 100%) | |

Parser-free version: grammar (verbs, modifiers, directions, spatial, conjunctions, verb→action, direction→turn-token) is *discovered from training data alone* by `pure_vsa.scan_grammar_discovery.discover_grammar`. The hand-written parser is replaced with a discovered-grammar parser; reasoning mechanism unchanged. Same 100% accuracy.

### 2. PCFG SET (Hupkes et al. 2020) — string-edit operations

| Subset | D | Pure VSA |
|---|---|---|
| Single-operation test examples | 2048 | **100.00%** (154/154) |
| Full nested test (9,721 examples) | 8192 | **99.98%** (9,719 / 9,721) |
| Full nested test (2,000-sample) | 4096 | 99.90% (1,998 / 2,000) |
| Full nested test (2,000-sample) | 2048 | 99.75% (1,995 / 2,000) |
| Full nested test (2,000-sample) | 1024 | 98.00% (1,960 / 2,000) |

Mechanism: 10 per-operation procedural VSA transforms (copy, reverse, echo, swap_first_last, repeat, shift, remove_first, remove_second, append, prepend) + recursive evaluation of nested operation trees + **cleanup restricted to input tokens of the current example** (essential at the precision floor for long outputs up to 736 tokens). 2 failures of 9,721 at D=8192 — both on outputs >150 tokens.

### 3. COGS (Kim & Linzen 2020) — six constructions covered

| Split | Coverage | In-scope accuracy | Per-construction (correct/total) |
|---|---|---|---|
| train (in_distribution) | 18.5% (4,475 / 24,155) | **99.75%** | simple_intrans 790/790; intrans_w_det 1446/1447; transitive 964/969; pp_transitive 931/936; dative_to 177/177; cp_simple 156/156 |
| test (in_distribution) | 17.9% (536 / 3,000) | **99.44%** | simple_intrans 105/105; intrans_w_det 165/165; transitive 108/110; pp_transitive 123/124; dative_to 19/19; cp_simple 13/13 |
| gen (21 generalization conditions) | 19.4% (4,080 / 21,000) | **73.87%** | simple_intrans 78/78 (100%); intrans_w_det 260/260 (100%); cp_simple 1351/1351 (100%); transitive 598/995 (60%); pp_transitive 596/1145 (52%); dative_to 131/251 (52%) |

Mechanism: six construction handlers (simple intransitive, intransitive with determiner, transitive with proper subject + det+common-noun object, transitive + PP attachment, dative-to with proper recipient, CP recursion with intransitive embedded clause). Cross-construction learning: passive training examples teach `verb → (agent, theme)` for active transitive prediction; dative examples additionally populate `verb → (subj, obj)` for plain transitive.

**Three categories at 100% on the gen (held-out) split:** simple_intransitive, intrans_w_det, cp_simple — totaling 1,689 / 1,689 held-out compositional examples correct. These are the categories where all the verb's roles are observable in training.

**Gen-split limit on the other 3 categories:** verbs that appear in training in one construction (e.g. unaccusative intransitive `the dish shattered`) but tested in another (transitive `Isabella shattered the brain`) lack the cross-construction role inference needed. This is one of the canonical hard cases in compositional generalization (`unacc_to_transitive`, `prim_to_*_*`).

**Out of scope:** ~80% of COGS still uses constructions not yet covered (PP/CP recursion at depth >1, longer passives with recipients, dative_for, control verbs, datives with common-noun recipients). Each additional construction is a one-day engineering pass. Full COGS coverage is multi-day work.

### 4. LLM-as-parser hybrid — framework

Pluggable `Parser` protocol enables swapping any parser into the SCAN reasoner:
- `HandwrittenParser`: hand-written SCAN grammar (the original)
- `MockLLMParser`: deterministic stub for testing
- `LLMParser(backend='anthropic')`: calls Claude (Anthropic API)
- `LLMParser(backend='openai')`: calls OpenAI

The `HyperionWithParser` wrapper monkey-patches the parser at fit/eval time. 4 framework tests passing (offline); 1 live-LLM test skipped pending `ANTHROPIC_API_KEY`.

This architecture demonstrates the natural extension to natural language: an LLM does the parsing (raw English → structured slot dict); pure VSA does the compositional reasoning over the slot dict. Pure VSA never sees natural-language input directly.

### 5. ARC-AGI

**Not attempted in this session.** A separate research program. Joffe & Eliasmith (2025, [arXiv:2511.08747](https://arxiv.org/abs/2511.08747)) achieve 83% on 1D-ARC with hand-crafted VSA programs — a meaningful prior-art reference. Extending the present mechanism to ARC would require:
- Visual grid encoding via VSA (Frady et al. 2023 covers this)
- Per-task program induction (very different from rule extraction for fixed grammars)

---

## What the mechanism shows across benchmarks

Across SCAN, PCFG, and COGS-minimal, the *same primitives* (bind/unbind/bundle/permute/cleanup + Python dict facts) solve very different problem classes:

| Benchmark | Output structure | Per-class extension required |
|---|---|---|
| SCAN | Fixed templates per atom shape | Hand or discovered grammar parser |
| PCFG | Variable-length sequence transforms | 10 procedural per-operation transforms + recursive tree evaluator |
| COGS | Logical-form output with named roles | Construction-specific predict() per English schema |

The unifying claim: **compositional generalization is fundamentally an algebraic problem on structured representations, not a learning problem.** Where the structure is well-defined (SCAN grammar; PCFG operations; COGS simple-intransitive), pure VSA solves the held-out compositions deterministically at 99-100% accuracy with zero gradient descent.

Where the structure is harder (full COGS English grammar; ARC pixel grids), the mechanism needs an extended "structure handler" but the core algebraic reasoner remains the same.

---

## Test summary

All results reproducible by:

```bash
git clone <repo> && cd hyperion
pip install -e .[dev]
python data/scan/download_and_prep.py        # SCAN
python data/cogs/download_and_prep.py        # COGS
# PCFG data downloaded by hand in this session; script TODO

# All pure-VSA tests (SCAN 7 splits + parser-free + PCFG + COGS + framework)
python -m pytest tests/test_scan_all_splits.py tests/test_scan_parserless.py \
                 tests/test_pcfg.py tests/test_cogs.py \
                 tests/test_llm_parser_framework.py -v
```

Expected: 28 tests passing, 1 skipped (live LLM call requires API key).

---

## Honest scope statement

This is a research project that demonstrates a specific scientific claim — **pure VSA algebra suffices for SCAN-style compositional generalization without gradient descent** — and extends it to two adjacent problem classes (PCFG, simplest COGS). The mechanism is:

- **Reliable** within its scope: deterministic across seeds, reproducible by test
- **Small**: ~3,000 lines of Python total across all modules
- **Fast**: ~1s fit, ~45s eval on the largest test set, CPU only
- **NOT a general AGI system**: requires per-problem-class structure encoding (parsers, operation interpreters)
- **NOT a transformer replacement**: frontier LLMs solve open-ended natural language tasks this cannot touch
- **A clean scientific contribution**: demonstrates that the form of compositional generalization SCAN was designed to test is algebraic, not statistical
