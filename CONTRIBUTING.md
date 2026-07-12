# Contributing to RAIN

This document sets expectations for contributors and serves as the canonical
reference for code, commit, and review standards.

## Before you write any code

1. Read `docs/RAIN-NET.md` and `docs/PIVOT.md` end-to-end. The architecture is
   tightly coupled; partial reads cause incoherent contributions.
2. Read `docs/architecture/fep-unified-objective.md` and
   `docs/architecture/benchmark-suite.md` to see how existing modules map to
   the objective and the acceptance benchmarks.

## Code standards

- **Python:** ruff + black, type-annotated, no `Any` in public APIs, MSRV
  Python 3.11. `from __future__ import annotations` everywhere.
- **Rust:** clippy + rustfmt clean, MSRV 1.85, no `unsafe` outside the FFI
  layer, every public symbol documented.
- **TypeScript:** prettier + eslint, strict mode, no `any` in non-test code.

## Branching

- `main` is protected. No direct commits.
- Feature branches: `feature/<short-kebab-name>`.
- Bug branches: `fix/<short-kebab-name>`.
- One pull request per atomic change. Squash-merge only.

## Commit messages

- Imperative mood. "Add X" not "Added X".
- Body explains WHY when not obvious.
- Reference the section of the relevant architecture doc that this commit
  advances when applicable: e.g. `Refs: docs/architecture/fep-unified-objective.md §3`.

## Pull request checklist

- [ ] Tests added for new logic. Pre-existing test suites must still pass.
- [ ] Tier-3 architectural-soundness checks (A1-A5 in
      `docs/architecture/benchmark-suite.md`) still green after the change.
- [ ] If the change touches a Tier-1 capability surface, the corresponding
      novelty benchmark (N1-N5) must be re-run and posted in the PR.
- [ ] No new `any` types (TypeScript), no `unsafe` outside FFI (Rust), no
      `Any` returns in public Python APIs.
- [ ] No `.env`, secrets, credentials, or unrelated personal tooling committed.
- [ ] CHANGELOG entry under `## [Unreleased]`.

## Testing protocol

- `pytest tests/` for Python unit + integration.
- `cargo test --workspace` for Rust.
- `pnpm test` inside `rain-kernel/` and `rain-chat/` for TS.
- `python evals/reproduce.py` for the full 18-benchmark Tier 1/2/3 sweep
  (slow, run on milestones).

## Code review

- All non-trivial PRs reviewed by a maintainer.
- Live end-to-end testing required before any UI-facing or end-to-end claim.

## Disagreements

If you and the reviewer disagree on direction:
1. Document the disagreement in the PR.
2. Reference the section of the relevant architecture doc that should resolve it.
3. If the architecture docs are silent, escalate to the maintainer for a
   documented amendment via a separate PR.
4. Do not merge contested PRs without explicit maintainer approval.

## Contact

Maintainer: Kristian Baer — `info@northtek.io`
