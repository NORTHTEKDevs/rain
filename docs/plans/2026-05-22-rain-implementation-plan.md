# RAIN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship `v0.1.0` of RAIN — a working, falsifiable instance of the
Resonant Active Inference Network on a single workstation (RTX 4090 + CPU),
passing all 18 Tier 1/2/3 acceptance benchmarks, with the LLM-equivalent
`rain-chat` UI and reproducible eval harness, in 12 weeks.

**Architecture:** Bipolar 10K-dim VSA state evolved by HYMN MLP, routed by
SOFAR frequency-banded SVD beam-steering, decoded by a 7-source EFE decoder,
learned by local rules (PCN + LSM-RLS + Tsetlin + FEP rank-1) at runtime,
backed by sharded HRR KB + crystals + ToM + self-model + calibration, evolved
by NSGA-II + champion/challenger, dispatched across WASM/Rust/Go/TS. See
`docs/plans/2026-05-22-rain-design.md` for the full design.

**Tech Stack:** Python 3.11+, Rust 2024 (PyO3 + wasm-bindgen), TypeScript
(reuse of `cognitive-kernel-polyglot`), SentencePiece BPE 32K, SQLite via
sqlx, Tauri 2 + React + shadcn for chat UI, GitHub Actions for CI.

---

## Required reading before any task

1. `docs/plans/2026-05-22-rain-design.md` — master design
2. `docs/architecture/component-map.md` — module sourcing
3. `docs/architecture/benchmark-suite.md` — acceptance contract
4. `VENDORED.md` — provenance protocol
5. `CRYSTAL.md` — project crystal

Use `superpowers:test-driven-development` for every new module. Use
`superpowers:systematic-debugging` for any failing test that isn't trivial.
Use `code-reviewer` before tagging any milestone.

---

## Phase 0 — Scaffold (COMPLETE, 2026-05-22)

Repository initialized at `~/projects/active/rain/` and pushed to
`https://github.com/NORTHTEKDevs/rain`. Tag `v0.0.0-scaffold`. Lockdown
applied (private, squash-only, Dependabot, no wiki/projects/discussions).
All design + architecture + benchmark + funding docs committed.

No further tasks in Phase 0.

---

## Phase 1 — Bootstrap setup (Week 1)

Goal: codebook + KB + tokenizer ready to feed HYMN pre-training.

### Task 1.1: Wire pyproject + dev deps + verify install

**Files:**
- Modify: `pyproject.toml` (only if needed during install)
- Create: `tests/test_smoke.py`

**Step 1: Write the failing smoke test**

```python
# tests/test_smoke.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Smoke test — imports work and version is exposed."""

def test_import_rain():
    import rain
    assert rain.__version__ == "0.0.0"

def test_python_version():
    import sys
    assert sys.version_info >= (3, 11)
```

**Step 2: Run to verify it fails on a fresh env**

```bash
cd ~/projects/active/rain
python -m venv .venv && source .venv/Scripts/activate  # Windows bash
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```

Expected: PASS (rain module imports, version exposed).

**Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add smoke import test for rain package"
```

### Task 1.2: Vendor RCK source — relational, knowledge_base, pcn, efe

**Files:**
- Create: `third_party/rck/relational.py` (vendored, original headers preserved)
- Create: `third_party/rck/knowledge_base.py`
- Create: `third_party/rck/pcn.py`
- Create: `third_party/rck/efe.py`
- Create: `third_party/rck/__init__.py`
- Modify: `VENDORED.md` — append provenance log entries

**Step 1: Compute provenance hashes and copy**

```bash
cd ~/projects/active/rain
for f in relational.py knowledge_base.py pcn.py efe.py; do
    SHA=$(sha256sum ~/projects/active/rck/rck/$f | awk '{print $1}')
    SRC_SHA=$(cd ~/projects/active/rck && git rev-parse HEAD)
    echo "$f: src=$SRC_SHA file_sha256=$SHA"
    cp ~/projects/active/rck/rck/$f third_party/rck/$f
done
touch third_party/rck/__init__.py
```

**Step 2: Append entries to `VENDORED.md`** with the hashes printed above.

**Step 3: Commit**

```bash
git add third_party/rck/ VENDORED.md
git commit -m "vendor: copy RCK core (relational, knowledge_base, pcn, efe)

Source: ~/projects/active/rck @ <SHA>
See VENDORED.md for per-file hashes."
```

### Task 1.3: Port relational.py to bipolar 10K-dim

**Files:**
- Create: `rain/core/relational.py`
- Create: `tests/core/test_relational.py`

**Step 1: Read the source**

Read `third_party/rck/relational.py` to understand the FHRR API (bind,
bundle, unbind, codebook).

**Step 2: Write failing tests for bipolar invariants**

```python
# tests/core/test_relational.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for bipolar 10K-dim VSA primitives."""

import numpy as np
import pytest
from rain.core.relational import Codebook, bind, bundle, unbind

D = 10000

def test_codebook_values_are_bipolar():
    cb = Codebook(vocab_size=100, dim=D, seed=42)
    hv = cb.vector("hello")
    assert hv.shape == (D,)
    assert set(np.unique(hv).tolist()) <= {-1, 1}

def test_bind_is_self_inverse_for_bipolar():
    cb = Codebook(vocab_size=10, dim=D, seed=0)
    a = cb.vector("a")
    b = cb.vector("b")
    bound = bind(a, b)
    recovered = unbind(bound, a)
    # Cosine similarity to b should be high
    sim = (recovered @ b) / (np.linalg.norm(recovered) * np.linalg.norm(b))
    assert sim > 0.95

def test_bundle_preserves_membership():
    cb = Codebook(vocab_size=10, dim=D, seed=0)
    items = [cb.vector(s) for s in ["a", "b", "c"]]
    bundle_hv = bundle(items)
    for item in items:
        sim = (bundle_hv @ item) / D
        assert sim > 0.3  # bundling preserves approximate membership

def test_seeding_is_deterministic():
    cb1 = Codebook(vocab_size=5, dim=D, seed=123)
    cb2 = Codebook(vocab_size=5, dim=D, seed=123)
    assert np.array_equal(cb1.vector("x"), cb2.vector("x"))
```

**Step 3: Run tests — verify fail**

```bash
pytest tests/core/test_relational.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rain.core.relational'`.

**Step 4: Write minimal bipolar implementation**

```python
# rain/core/relational.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/relational.py — adapted from 4K FHRR to 10K bipolar
"""Bipolar 10K-dim VSA primitives. Bind = element-wise product. Bundle = sum + sign."""

from __future__ import annotations

import hashlib
import numpy as np

DEFAULT_DIM = 10000


def _seed_hash(name: str, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


class Codebook:
    """Deterministic hypervector codebook over a vocabulary."""

    def __init__(self, vocab_size: int, dim: int = DEFAULT_DIM, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}

    def vector(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        rng = np.random.default_rng(_seed_hash(name, self.seed))
        v = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
        self._cache[name] = v
        return v


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bipolar bind = element-wise product. Self-inverse."""
    return (a * b).astype(np.int8)


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Bipolar unbind = bind (because product is self-inverse over {-1, +1})."""
    return bind(bound, key)


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Bipolar bundle = element-wise sum then sign."""
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.sign(summed + (summed == 0)).astype(np.int8)  # break ties to +1
```

**Step 5: Run tests — verify pass**

```bash
pytest tests/core/test_relational.py -v
```

Expected: 4 PASS.

**Step 6: Commit**

```bash
git add rain/core/relational.py tests/core/test_relational.py
git commit -m "feat(core): bipolar 10K-dim VSA primitives (bind/bundle/unbind/codebook)

Refs: docs/plans/2026-05-22-rain-design.md §2 (binding convention locked)"
```

### Task 1.4: Port and test sharded HRR KB

**Files:**
- Create: `rain/core/knowledge_base.py`
- Create: `tests/core/test_knowledge_base.py`

**Step 1: Write failing tests**

```python
# tests/core/test_knowledge_base.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for sharded HRR knowledge base."""

import numpy as np
import pytest
from rain.core.knowledge_base import ShardedKB

def test_write_then_read():
    kb = ShardedKB(num_shards=8, dim=10000, seed=0)
    kb.write("rome", "capital_of", "italy")
    assert kb.query("rome", "capital_of") == "italy"

def test_capacity_scales_with_shards():
    kb = ShardedKB(num_shards=64, dim=10000, seed=0)
    for i in range(1000):
        kb.write(f"s{i}", "r", f"o{i}")
    correct = sum(1 for i in range(1000) if kb.query(f"s{i}", "r") == f"o{i}")
    assert correct / 1000 >= 0.85

def test_unknown_returns_none():
    kb = ShardedKB(num_shards=8, dim=10000, seed=0)
    assert kb.query("paris", "capital_of") is None
```

**Step 2: Run, verify fail.**

**Step 3: Implement** by adapting `third_party/rck/knowledge_base.py` to use
the new bipolar primitives from `rain.core.relational`. Keep the blake2b
shard-routing approach unchanged.

**Step 4: Run, verify pass.**

**Step 5: Commit**

```bash
git add rain/core/knowledge_base.py tests/core/test_knowledge_base.py
git commit -m "feat(core): sharded HRR knowledge base over bipolar primitives"
```

### Task 1.5: BPE tokenizer wrapper

**Files:**
- Create: `rain/tokenize/bpe.py`
- Create: `tests/tokenize/test_bpe.py`

**Step 1: Write failing tests**

```python
# tests/tokenize/test_bpe.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import pytest
from rain.tokenize.bpe import BPETokenizer

def test_train_and_encode():
    tok = BPETokenizer(vocab_size=512)
    tok.train(["the quick brown fox", "the lazy dog"])
    ids = tok.encode("the fox")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

def test_round_trip():
    tok = BPETokenizer(vocab_size=512)
    tok.train(["hello world", "goodbye world"])
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids).strip() == text
```

**Step 2: Run, verify fail.**

**Step 3: Implement** as a thin wrapper around `sentencepiece`:

```python
# rain/tokenize/bpe.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""BPE tokenizer via SentencePiece. Default vocab 32K, configurable."""

from __future__ import annotations
import io
import tempfile
from pathlib import Path
import sentencepiece as spm


class BPETokenizer:
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size
        self._sp: spm.SentencePieceProcessor | None = None

    def train(self, texts: list[str]) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for t in texts:
                f.write(t + "\n")
            path = f.name
        model_buf = io.BytesIO()
        spm.SentencePieceTrainer.train(
            input=path,
            model_writer=model_buf,
            vocab_size=self.vocab_size,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        self._sp = spm.SentencePieceProcessor(model_proto=model_buf.getvalue())

    def encode(self, text: str) -> list[int]:
        assert self._sp is not None, "tokenizer not trained"
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        assert self._sp is not None, "tokenizer not trained"
        return self._sp.decode(ids)

    def save(self, path: str | Path) -> None:
        assert self._sp is not None
        Path(path).write_bytes(self._sp.serialized_model_proto())

    def load(self, path: str | Path) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(path))
```

**Step 4: Run, verify pass.**

**Step 5: Commit**

```bash
git add rain/tokenize/bpe.py tests/tokenize/test_bpe.py
git commit -m "feat(tokenize): BPE tokenizer wrapper (SentencePiece, 32K default)"
```

### Task 1.6: Codebook warm-start from pretrained embeddings

**Files:**
- Create: `scripts/bootstrap_warm_start.py`
- Create: `tests/test_bootstrap_warm_start.py`

**Step 1: Write failing test**

```python
# tests/test_bootstrap_warm_start.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
from rain.core.relational import Codebook
from scripts.bootstrap_warm_start import warm_start_from_vectors

def test_warm_start_overrides_random():
    cb = Codebook(vocab_size=3, dim=10000, seed=0)
    original = cb.vector("hello").copy()
    fake_vectors = {"hello": np.random.RandomState(99).randn(300)}
    warm_start_from_vectors(cb, fake_vectors)
    new = cb.vector("hello")
    # Same set of -1/+1
    assert set(np.unique(new).tolist()) <= {-1, 1}
    # Different from random init
    assert not np.array_equal(new, original)
```

**Step 2: Run, verify fail.**

**Step 3: Implement projection** from real-valued vectors to bipolar via sign + tile-and-truncate:

```python
# scripts/bootstrap_warm_start.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Warm-start the codebook from pretrained embeddings via sign projection.

Phase 1.1 of the training protocol. Not an LLM substrate at inference —
just smart init."""

from __future__ import annotations
import numpy as np
from rain.core.relational import Codebook


def warm_start_from_vectors(cb: Codebook, vectors: dict[str, np.ndarray]) -> int:
    """Tile each input vector to reach `cb.dim`, take sign, store. Returns count loaded."""
    loaded = 0
    for name, v in vectors.items():
        repeats = (cb.dim + len(v) - 1) // len(v)
        tiled = np.tile(v, repeats)[: cb.dim]
        bipolar = np.sign(tiled + (tiled == 0)).astype(np.int8)
        cb._cache[name] = bipolar
        loaded += 1
    return loaded
```

**Step 4: Run, verify pass.**

**Step 5: Commit**

```bash
git add scripts/bootstrap_warm_start.py tests/test_bootstrap_warm_start.py
git commit -m "feat(bootstrap): warm-start codebook from pretrained embeddings via sign proj"
```

### Task 1.7: KB seeding script

**Files:**
- Create: `scripts/seed_kb.py`
- Create: `tests/test_seed_kb.py`
- Create: `data/seed_facts.jsonl` (small sample for tests)

**Step 1: Sample seed facts**

```
# data/seed_facts.jsonl
{"s": "rome", "r": "capital_of", "o": "italy"}
{"s": "paris", "r": "capital_of", "o": "france"}
{"s": "water", "r": "boils_at", "o": "100C"}
```

**Step 2: Write failing test**

```python
# tests/test_seed_kb.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from rain.core.knowledge_base import ShardedKB
from scripts.seed_kb import seed_from_jsonl

def test_seed_loads_facts(tmp_path):
    p = tmp_path / "facts.jsonl"
    p.write_text('{"s":"a","r":"r","o":"b"}\n')
    kb = ShardedKB(num_shards=4, dim=10000, seed=0)
    n = seed_from_jsonl(kb, str(p))
    assert n == 1
    assert kb.query("a", "r") == "b"
```

**Step 3: Run, verify fail. Step 4: Implement. Step 5: Run, verify pass. Step 6: Commit.**

```python
# scripts/seed_kb.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

from __future__ import annotations
import json
from rain.core.knowledge_base import ShardedKB


def seed_from_jsonl(kb: ShardedKB, path: str) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fact = json.loads(line)
            kb.write(fact["s"], fact["r"], fact["o"])
            n += 1
    return n
```

```bash
git add scripts/seed_kb.py data/seed_facts.jsonl tests/test_seed_kb.py
git commit -m "feat(data): KB seed loader from JSONL"
```

### Task 1.8: Week-1 integration test

**Files:**
- Create: `tests/integration/test_phase1_setup.py`

Confirms that warm-start + KB seed + tokenizer all compose end-to-end on a
tiny corpus.

```python
# tests/integration/test_phase1_setup.py
# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np
from rain.core.relational import Codebook
from rain.core.knowledge_base import ShardedKB
from rain.tokenize.bpe import BPETokenizer
from scripts.bootstrap_warm_start import warm_start_from_vectors

def test_end_to_end_phase1_setup():
    tok = BPETokenizer(vocab_size=256)
    tok.train(["the quick brown fox", "the lazy dog runs"])

    cb = Codebook(vocab_size=256, dim=10000, seed=42)
    fake_vecs = {"hello": np.random.RandomState(0).randn(300)}
    n = warm_start_from_vectors(cb, fake_vecs)
    assert n == 1

    kb = ShardedKB(num_shards=8, dim=10000, seed=42)
    kb.write("dog", "is_a", "animal")
    assert kb.query("dog", "is_a") == "animal"
```

**Run:** `pytest tests/integration -v`. Expected: PASS.

**Commit:**

```bash
git add tests/integration/test_phase1_setup.py
git commit -m "test(integration): Phase 1 setup end-to-end (tokenizer + warm-start + KB)"
git tag v0.0.1-phase1-setup
```

---

## Phase 2 — HYMN gradient pre-train on Tiny Shakespeare (Weeks 2-3)

Goal: HYMN MLP spine trained on Tiny Shakespeare, hits Tier 2 L1 val-loss
≤ 1.55 nats/char, weights frozen at end of phase.

### Task 2.1: Vendor Hyperion HYMN code

**Files:**
- Create: `third_party/hyperion/hymn/<files>`
- Create: `third_party/hyperion/vsa_core/<files>`
- Modify: `VENDORED.md`

Copy + provenance hashes + log entry. **Commit:** `vendor: copy Hyperion HYMN
+ vsa_core for adaptation`.

### Task 2.2: Port HYMN MLP forward to Rust hot path

**Files:**
- Modify: `rain-rs/src/hymn.rs`
- Modify: `rain-rs/Cargo.toml` (enable `pyo3` feature for Python binding)
- Create: `rain-rs/tests/test_hymn.rs`
- Create: `rain/spine.py`

Tasks broken further when reached:
- 2.2a: Define `HymnConfig` (in_dim, hidden_dim, out_dim, activation, dropout).
- 2.2b: Implement `forward(state: &[i8], input: &[i8]) -> Vec<i8>` returning bipolar output.
- 2.2c: Add PyO3 binding `rain_rs.hymn_forward(state_arr, input_arr)`.
- 2.2d: Benchmark via `criterion`. Target: <500 μs / forward at D=10K, H=4K on workstation CPU.
- 2.2e: Python wrapper in `rain/spine.py` calls into Rust.

Each substep is one TDD cycle. Commit after each green test.

### Task 2.3: Bootstrap pre-training driver

**Files:**
- Create: `scripts/pretrain_hymn.py`
- Create: `tests/test_pretrain_hymn_smoke.py`

Smoke test trains for 10 steps on a 1KB corpus and asserts loss decreased.
Full training run is invoked manually (not in CI) with target Tiny Shakespeare
val-loss ≤ 1.55. Track via `tensorboard` (optional dep).

### Task 2.4: Freeze + checkpoint HYMN weights

**Files:**
- Modify: `scripts/pretrain_hymn.py` — add `--save-frozen` mode

Outputs `.npz` checkpoint with HYMN weights + sidecar metadata. Frozen flag
prevents Phase 2 from overwriting.

### Task 2.5: Tier 2 L1 benchmark

**Files:**
- Create: `evals/tier2_llm_parity/tiny_shakespeare.py`
- Create: `evals/baselines/nanogpt/` (vendored nanoGPT for matched-FLOPs baseline)

Acceptance: produces a JSON result with `{"val_loss": <number>, "pass": <bool>}`
where pass is val_loss ≤ 1.55. Run on workstation. Commit results to
`evals/results/<date>/tier2/L1.json`.

Tag `v0.0.2-hymn-pretrained` after L1 passes.

---

## Phase 3 — Remaining Phase 1 components (Week 4)

Goal: SOFAR adapter, Tsetlin seeding, LSM init, FEP A init all operational.

### Task 3.1: SOFAR routing transplant — mapper (SVD)

Vendor `sofar/mapper.py`. Adapt to operate on `[Codebook, Roles, W_HYMN_in]`
instead of `[W_q, W_k, W_v, ...]`. Tests cover SVD shape correctness + cached
recomputation trigger.

### Task 3.2: SOFAR routing — encoder (Hadamard partition)

Adapt `sofar/encoder.py`. FFT → Walsh-Hadamard. Tests cover round-trip
reconstruction (`s = s^L + s^M + s^H`) within ±1 in bipolar sign-space.

### Task 3.3: SOFAR routing — beam (FOCUS/SWEEP/TRACK envelopes)

Adapt `sofar/attention.py`. LoRA adapter rank-projected onto routing
directions. Tests cover identity-at-init invariance + energy normalization.

### Task 3.4: Tsetlin clause seeding

Vendor `rck/tsetlin.py`. Port hot path to Rust (`rain-rs/src/tsetlin.rs`)
during Phase 4 if benchmarking shows Python bottleneck. Run Tsetlin Type-I
training on Tiny Shakespeare for clause-population init.

### Task 3.5: LSM RLS init

Vendor `rck/liquid_state.py`. Run RLS single-pass over the same Tiny
Shakespeare corpus.

### Task 3.6: FEP rank-1 A matrix init from PCA

Vendor `rck/fep.py`. PCA over the HYMN residual stream captured during
Phase 2 → low-rank A initialization.

### Task 3.7: Phase-1 integration

Compose all of the above behind `rain.train.bootstrap.bootstrap_phase1()`.
Single entry point. Runs all 7 sub-steps (1.1-1.7 per design doc).

Tag `v0.0.3-phase1-complete`.

---

## Phase 4 — Tier 3 architectural soundness (Week 5)

Goal: A1-A5 all green.

### Task 4.1: A1 — EFE source-mix sanity

Vendor `rck/efe.py`, extend from 4-source to 7-source fusion per design doc
§ 3 Step 5. Add `evals/tier3_soundness/efe_source_mix.py` running 100-prompt
probe and asserting each source contributes ≥ 5%.

### Task 4.2: A2 — SOFAR-on-VSA ablation

`evals/tier3_soundness/sofar_ablation.py` runs val-loss with and without
routing. Threshold: ≥ 3% improvement. **Kill trigger:** if fails, drop SOFAR
routing module from active path; document in CHANGELOG.

### Task 4.3: A3 — NSGA-II promotion rate

Vendor polyglot kernel `evolution.rs` / `evolution.ts`. Run 1000-gen
background evolution on a HYMN-MLP-variant population. Assert ≥ 1 promotion
per 100 gens. Bench: `evals/tier3_soundness/nsga2_promotion.py`.

### Task 4.4: A4 — Local-rule continual stability

Implement Phase 2 local-rule training loop. Run 10K turns of synthetic
dialogue + fact-teaching. Assert no NaN, ECE drift ≤ 0.1, no KB corruption.

### Task 4.5: A5 — Polyglot dispatch correctness

Verify WASM/Rust/Go/TS produce bit-identical outputs (per polyglot kernel's
existing tests; just re-run inside RAIN's eval harness).

Tag `v0.0.4-tier3-green`.

---

## Phase 5 — Tier 1 novelty surfaces (Weeks 6-7)

Goal: N1-N5 all green.

### Task 5.1: N1 — Continual learning A→B→A retention

Vendor `rck/conscious_agent.py` and run RCK's existing retention test under
the RAIN integrated model. Tighten threshold to 0.5.

### Task 5.2: N2 — Calibrated uncertainty

Vendor `rck/metacog.py` + reconcile with polyglot kernel `calibration.ts`.
1000-question probe across 14 relations. ECE ≤ 0.05.

### Task 5.3: N3 — Compositional generalization

Vendor `rck/compose.py` + Hyperion's `eval_scan.py`. Run SCAN add-primitive
and 3/4/5-slot composition probes. RCK v1.1 already passes; just re-verify
under RAIN integrated model.

### Task 5.4: N4 — Structural ToM

Vendor `rck/theory_of_mind.py`. Add 10 Sally-Anne variants and 5 20-turn
BDI dialogues. Pass: 10/10 Sally-Anne + 18/20 BDI.

### Task 5.5: N5 — Grounded transparency

Vendor `rck/explain.py` + `rck/think_aloud.py`. 100-claim audit. Pass: ≥98%
citation coverage + ≤1% hallucination.

Tag `v0.0.5-tier1-green`.

---

## Phase 6 — Tier 2 LLM parity at toy scale (Week 8)

Goal: L1-L8 all green.

Implement each benchmark per `docs/architecture/benchmark-suite.md` § "Tier 2".
Run RWKV-7-mini + LFM2-small + Mamba-3-small + Pythia-160M baselines under
matched-FLOPs settings in `evals/baselines/`.

Tag `v0.0.6-tier2-green`.

---

## Phase 7 — Ten capability additions (Weeks 9-10)

Goal: all 10 additions from design doc § 3 integrated and tested.

One task each:
- 7.1: KB-RAG-as-decode-signal (`rain/additions/kb_rag.py`)
- 7.2: Native thinking mode (`rain/additions/thinking_mode.py`)
- 7.3: Speculative decoding (`rain/additions/speculative.py`)
- 7.4: Persistent agent identity (`rain/additions/identity.py`)
- 7.5: Federated bundle primitive (`rain/additions/federated.py`)
- 7.6: Sleep/replay consolidation (`rain/additions/sleep_replay.py`)
- 7.7: Self-verification head (`rain/additions/self_verify.py`)
- 7.8: Tool registry (`rain/additions/tool_registry.py`)
- 7.9: Values layer (`rain/additions/values.py`)
- 7.10: Multimodal stub (`rain/additions/multimodal_stub.py`)

Each: TDD cycle, integration test, commit. Tag `v0.0.7-additions-complete`.

---

## Phase 8 — Polish + chat UI + v0 tag (Weeks 11-12)

### Task 8.1: `rain-cli` REPL + batch eval

CLI entry points wired up: `rain serve`, `rain eval --tier all`, `rain repl`.

### Task 8.2: `rain-chat` Tauri UI

LLM-equivalent chat surface. Confidence chips, citation toggle, "RAIN learned
X this turn" banners. Matches Northtek design system (accent `#0A84FF`,
custom SVG icons, dark-mode default).

### Task 8.3: 5-minute demo video script + recording

Demonstrates all 5 Tier-1 surfaces in one session: continual learning,
calibrated refusal, compositional novelty, ToM story, grounded explanation.

### Task 8.4: README polish + final eval sweep

`python evals/reproduce.py --tier all` produces all 18 benchmark results.
Confirm pass. README updated with the result table.

### Task 8.5: code-reviewer + production-certifier pass

Run `code-reviewer` skill on the full repo. Resolve all findings. Run
`production-certifier` for the v0 ship gate. Target: 9/10+ cert score.

### Task 8.6: Tag v0.1.0

```bash
git tag v0.1.0
git push origin v0.1.0
```

Update CHANGELOG with the v0 release notes. Cross-commit a copy of the
master design doc + final benchmark results to a private internal archive
(in case of patent priority disputes).

---

## Cross-cutting practices

- **TDD always.** Use `superpowers:test-driven-development`. Every module
  starts with a failing test.
- **Frequent commits.** Each green test → commit. Squash on merge.
- **Use the skill family:**
  - `superpowers:systematic-debugging` for any failing test that isn't
    trivial to fix.
  - `code-reviewer` after every Phase milestone.
  - `production-certifier` before each tagged release.
  - `generalization-probe` after each Tier-1 surface passes — stress-test
    with adversarial inputs.
- **No public commits.** Repo is private. No benchmarks posted anywhere
  external until patent filings are in.
- **Run header audit before every commit:**

  ```bash
  for f in $(git diff --cached --name-only | grep -E '\.(py|rs|ts)$'); do
      head -3 "$f" | grep -q "CONFIDENTIAL - PATENT PENDING" || echo "MISSING: $f"
  done
  ```

- **Update CHANGELOG.md every PR** under `[Unreleased]`.
- **Vendoring discipline.** Every copy goes through `VENDORED.md`
  with provenance hash. No vendored file imports anything outside RAIN's
  approved component set.

---

## Definition of done

Repository tagged `v0.1.0` with:
- All 18 Tier 1/2/3 benchmarks green on a fresh `python evals/reproduce.py`.
- `rain-chat` runs and renders the 5 Tier-1 capability surfaces interactively.
- 5-minute demo video produced and stored on encrypted backup.
- `code-reviewer` final pass clean.
- `production-certifier` ≥ 9/10.
- CHANGELOG complete.
- Patent provisional applications filed (or filing scheduled within 30 days).
- Internal stakeholder demo conducted.
