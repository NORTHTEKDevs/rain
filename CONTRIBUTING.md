# Contributing to RAIN

**This repository is currently solo-development under closed access.** This
document sets expectations for future authorized collaborators and serves as
the canonical reference for code, commit, and review standards.

## Before you write any code

1. Read `docs/plans/2026-05-22-rain-design.md` end-to-end. The architecture is
   tightly coupled; partial reads cause incoherent contributions.
2. Read `docs/architecture/component-map.md` to see which existing module is
   being adapted vs which is new.
3. Read `PATENT_NOTES.md`. Every source file you touch must keep the
   `CONFIDENTIAL - PATENT PENDING` header.
4. Confirm you have signed a confidentiality agreement and have been added to
   the NORTHTEKDevs/rain repository explicitly.

## Code standards

- **Python:** ruff + black, type-annotated, no `Any` in public APIs, MSRV
  Python 3.11. `from __future__ import annotations` everywhere.
- **Rust:** clippy + rustfmt clean, MSRV 1.85, no `unsafe` outside the FFI
  layer, every public symbol documented.
- **TypeScript:** prettier + eslint, strict mode, no `any` in non-test code.
- **Headers:** every source file starts with the copyright + patent-pending
  notice. See `scripts/check_headers.py` (added later).

## Branching

- `main` is protected. No direct commits.
- Feature branches: `feature/<short-kebab-name>`.
- Bug branches: `fix/<short-kebab-name>`.
- One pull request per atomic change. Squash-merge only.

## Commit messages

- Imperative mood. "Add X" not "Added X".
- Body explains WHY when not obvious.
- Reference the section of the design doc that this commit advances when
  applicable: e.g. `Refs: docs/plans/2026-05-22-rain-design.md §3.5`.
- No `Co-authored-by:` lines for AI assistants in the public-facing commit
  history. Internal review notes belong in the PR description, not the commit.

## Pull request checklist

- [ ] Tests added for new logic. Pre-existing 121-test RCK suite + 42-test
      SOFAR suite + integration tests must still pass.
- [ ] Tier-3 architectural-soundness checks (A1-A5 in the design doc) still
      green after the change.
- [ ] If the change touches a Tier-1 capability surface, the corresponding
      novelty benchmark (N1-N5) must be re-run and posted in the PR.
- [ ] No new `any` types (TypeScript), no `unsafe` outside FFI (Rust), no
      `Any` returns in public Python APIs.
- [ ] Headers preserved on every touched file.
- [ ] No `.env`, secrets, credentials, or unrelated personal tooling committed.
- [ ] CHANGELOG entry under `## [Unreleased]`.

## Testing protocol

- `pytest tests/` for Python unit + integration.
- `cargo test --workspace` for Rust.
- `pnpm test` inside `rain-kernel/` and `rain-chat/` for TS.
- `python evals/reproduce.py` for the full 18-benchmark Tier 1/2/3 sweep
  (slow, run on milestones).

## Code review

- All non-trivial PRs reviewed by code-reviewer agent + human reviewer.
- Production-certifier required before any `v0.x` tag.
- Live-flow-tester required before any UI-facing or end-to-end claim.

## Disagreements

If you and the reviewer disagree on direction:
1. Document the disagreement in the PR.
2. Reference the section of the design doc that should resolve it.
3. If the design doc is silent, escalate to the maintainer for a design-doc
   amendment via a separate PR.
4. Do not merge contested PRs without explicit maintainer approval.

## Contact

Maintainer: Kristian Baer — `info@northtek.io`
