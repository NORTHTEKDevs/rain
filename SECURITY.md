# Security Policy

**RAIN — Resonant Active Inference Network**
**CONFIDENTIAL — PATENT PENDING**

## Vulnerability reporting

Report suspected vulnerabilities privately to `info@northtek.io` with subject
prefix `[RAIN-SEC]`. Do not file public issues, do not post on social media,
and do not discuss in any non-NORTHTEKDevs channel until coordinated
disclosure has been agreed.

Acknowledgement target: 48 hours. Triage target: 5 business days.

## Threat model

This repository is a private research artifact. The threat model assumes:

- **Trusted developers** with signed CLAs operating on NORTHTEKDevs hardware.
- **Untrusted public** with no inbound access. The repo is private.
- **Untrusted dependency supply chain.** Dependabot + dependency review run
  weekly; transitive vulns triaged with elevated priority for any package in
  the cryptographic, networking, or serialization layers.
- **Untrusted training data.** Phase 1 corpora are vetted; Phase 2 continual
  learning writes are validated and tenant-scoped via the polyglot kernel's
  guardrail layer.

## IP rules in scope of security policy

- **No source on personal devices** outside the controlled workstation.
- **No source in third-party AI assistants** beyond Claude Code under
  NORTHTEKDevs operational rules (which run locally and do not exfiltrate).
- **No commits authored by unauthorized identities.** All commits must be
  signed by NORTHTEKDevs-authorized git identities.
- **No deploy to public infra** without explicit IP-clearance and counsel
  sign-off.

## Out of scope

- Hypothetical attacks against capabilities that are not yet implemented.
- Theoretical vulnerabilities without a working proof-of-concept.
- Issues in third-party dependencies that the upstream vendor has not yet
  addressed; we will track but not author fixes for these.

## Security-sensitive code paths

- `rain/data/bulk_ingest.py` — KB ingestion path. Inputs must be validated;
  triple extraction must not allow injection of arbitrary HRR bindings into
  privileged shards.
- `rain/additions/tool_registry.py` — tool dispatch. All tool calls go through
  the polyglot kernel's adapter layer with allowlist enforcement.
- `rain/session.py` — persistence. Path traversal must be impossible; all
  saved-state paths validated to live inside the configured store directory.
- `rain-rs/src/ffi/` — language boundaries. Memory safety enforced via Rust
  invariants; PyO3 + wasm-bindgen must never expose unchecked buffers to
  Python or JavaScript.

## Coordinated disclosure window

90 days from acknowledged report to public disclosure, extended by mutual
agreement if architectural changes are required.
