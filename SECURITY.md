# Security Policy

**RAIN — Resonant Active Inference Network**

## Vulnerability reporting

Report suspected vulnerabilities privately to `info@northtek.io` with subject
prefix `[RAIN-SEC]`. Please do not file public issues for undisclosed
vulnerabilities until we've had a chance to triage and coordinate a fix.

Acknowledgement target: 48 hours. Triage target: 5 business days.

## Threat model

This is an open-source research project. The threat model assumes:

- **Untrusted dependency supply chain.** Dependabot + dependency review run
  weekly; transitive vulns triaged with elevated priority for any package in
  the cryptographic, networking, or serialization layers.
- **Untrusted training data.** Any bulk-ingest or continual-learning paths
  must validate and scope inputs before they reach privileged state.
- **Untrusted public input** on any code path that consumes external data
  (ingestion, tool dispatch, session persistence).

## Out of scope

- Hypothetical attacks against capabilities that are not yet implemented.
- Theoretical vulnerabilities without a working proof-of-concept.
- Issues in third-party dependencies that the upstream vendor has not yet
  addressed; we will track but not author fixes for these.

## Security-sensitive code paths

- `rain/data/bulk_ingest.py` — KB ingestion path. Inputs must be validated;
  triple extraction must not allow injection of arbitrary HRR bindings into
  privileged shards.
- `rain/additions/tool_registry.py` — tool dispatch. All tool calls should go
  through an adapter layer with allowlist enforcement.
- `rain/session.py` — persistence. Path traversal must be impossible; all
  saved-state paths validated to live inside the configured store directory.
- `rain-rs/src/ffi/` — language boundaries. Memory safety enforced via Rust
  invariants; PyO3 + wasm-bindgen must never expose unchecked buffers to
  Python or JavaScript.

## Coordinated disclosure window

90 days from acknowledged report to public disclosure, extended by mutual
agreement if architectural changes are required.
