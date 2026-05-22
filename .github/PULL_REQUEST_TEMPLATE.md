# Pull Request

## Summary

<!-- One-paragraph description of what this PR does and why. -->

## Reference

<!-- Section of the design doc this PR implements. Required. -->

- Design doc: `docs/plans/2026-05-22-rain-design.md` § ___
- Component map: `docs/architecture/component-map.md` (if applicable)
- Vendoring entry: `VENDORED.md` (if vendoring new source)

## Checklist

- [ ] Tests added or updated
- [ ] `pytest -q` passes locally
- [ ] `cargo test --workspace` passes locally (if Rust touched)
- [ ] `ruff check` + `black --check` pass (if Python touched)
- [ ] `cargo fmt --check` + `cargo clippy -- -D warnings` pass (if Rust touched)
- [ ] Confidential header on every new file
- [ ] No `.env`, secrets, credentials, or unrelated personal tooling
- [ ] CHANGELOG entry under `[Unreleased]`
- [ ] No new `Any` in public Python APIs; no `any` in non-test TypeScript;
      no `unsafe` outside the FFI layer in Rust
- [ ] If this PR claims a Tier-1 capability surface (N1-N5) complete: the
      relevant benchmark in `evals/tier1_novelty/` has been re-run and the
      results posted below
- [ ] If this PR touches Tier-3-relevant code: all of Tier 3 (A1-A5) re-run

## Benchmarks (if applicable)

<!-- Paste the numeric results from any re-run benchmarks. -->

## Notes for reviewer

<!-- Anything reviewer should know — known gaps, alternative approaches
considered, open questions. -->
