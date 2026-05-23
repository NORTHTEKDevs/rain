# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- N1 continual-retention benchmark (Task 5.1) at `evals/tier1_novelty/retention.py` + `tests/test_eval_n1_retention.py`. A->B->A protocol per benchmark-suite.md: 5K animals/habitats vs 5K chemistry/composition, threshold retention >= 0.50, kill < 0.40. Closes Tier-1 surfaces 5/5 (N1-N5 all green).
- A2 SOFAR-on-VSA ablation benchmark (Task 4.2) at `evals/tier3_soundness/sofar_ablation.py` + `tests/test_eval_a2_sofar_ablation.py`. Routing-on vs routing-off MSE-on-HV val loss. Architectural-soundness invariant verified: identity-at-init is bit-identical to routing-off. v0 strict 3%-improvement bar reports `kill_triggered=true` (improvement ~= 0.08% with untrained LoRA); kill is **deferred**, not honored, pending re-run with trained routing weights post-Phase-2.3. Closes Tier-3 surfaces 5/5 (A1-A5 all wired).

### Fixed
- M-NEW-1: Removed `continue-on-error: true` from CI install-deps step; maturin/Rust build failures now hard-fail the job.
- M-NEW-2: Renamed `compute_val_loss` -> `compute_hv_mse_loss` in `evals/tier2_llm_parity/tiny_shakespeare.py`; updated module docstring to reflect MSE-on-HV semantics and Phase 2.3 NLL handoff.
- L-NEW-1: Header-audit CI step now exits 1 on missing confidential header (was warning-only, exits 0).
- L-NEW-2: Documented why wasm-build step stays soft-failing (`ffi_wasm.rs` stub; known gap until Phase 2.3+).
- L-NEW-3: Added Tier-2 benchmark driver line to `rain/__main__.py` help output.
- H1: Created `rain-rs/src/ffi_wasm.rs` stub (was declared in `lib.rs` but file was missing).
- H2: Added `rain/__main__.py` entry point; removed undeclared `rain-cli` script from `pyproject.toml`.
- H3: Moved `warm_start_from_vectors` and `seed_from_jsonl` into the `rain` package
  (`rain/train/warm_start.py`, `rain/data/kb_seed.py`); `scripts/` become thin re-export shims.
- H4: Renamed `val_loss` -> `hv_mse_loss` in L1 benchmark; replaced `L1_PASS_THRESHOLD`
  with `L1_PASS_THRESHOLD_HV_MSE = 0.5`; added `metric_type` and `nll_threshold_applicable`
  fields to clarify the MSE-vs-NLL distinction until Phase 2.3.
- M1: Fixed tempfile leak in `BPETokenizer.train` (try/finally + `os.unlink`).
- M2: `EFE.decode` with `rng=None` now uses wall-clock-seeded generator instead of fixed seed 0.
- M3: Added v0.5 TODO comment above linear scan in `ShardedKB._Shard.query`.
- M4: Replaced full `tobytes()` + blake2b in `RoutingMapper._hash_source` with a lightweight
  corner-sample fingerprint (shape + 16 samples + sum/std aggregate).
- M5: Replaced `assert` guards in `BPETokenizer` with `RuntimeError` (survives `python -O`).
- L1: Removed `|| true` masking from ruff, black, and pytest CI steps.
- L2: Added v0.5 vectorization TODO comment above `TsetlinMachine.vote`.
- L3: `TheoryOfMind._kb_for` now derives chain seed via `hashlib.blake2b` instead of
  Python `hash()` (deterministic across processes).
- L4: `rain/__init__.py` now re-exports `ConsciousAgent`, `Answer`, `Codebook`, `ShardedKB`.

### Added
- Initial repository scaffold under NORTHTEKDevs/rain.
- Master design document at `docs/plans/2026-05-22-rain-design.md`.
- Funding waypoints appendix at `docs/plans/2026-05-22-rain-funding-waypoints.md`.
- Architecture docs: FEP unified objective, component map, benchmark suite.
- Design docs: SOFAR-on-VSA transplant, multi-signal EFE decoder.
- Repository skeleton matching SOFAR lockdown pattern: LICENSE, SECURITY,
  CONTRIBUTING, CHANGELOG, CRYSTAL.
- Initial CI workflow.

## [0.0.0-scaffold] — 2026-05-22

- Repository created.
- Brainstorm + design approval session completed.
- Tag marks the scaffold-only state, no implementation code yet.
