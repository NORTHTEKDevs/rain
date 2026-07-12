//
// RAIN Rust hot-kernels crate.
//
// Modules:
//   vsa     - bipolar 10K-dim bind / bundle / unbind
//   hymn    - sequence-engine MLP forward
//   tsetlin - vectorised clause batch
//   sofar   - SVD over codebook + role matrices, beam-steering envelopes
//   nsga2   - non-dominated sorting + crowding distance + Pareto generation
//
// Stubs only at scaffold time. Implementation lands during the writing-plans
// implementation phase. See VENDORED.md for source-of-truth porting plan.

#![warn(unsafe_code)]
// missing_docs demoted to allow for v0 -- we will re-enable once the
// public Rust API stabilizes. Tracked in docs/READY_TO_SCALE.md.
#![allow(missing_docs)]
// Numeric kernels intentionally use integer-index loops (HYMN MLP, SVD,
// Tsetlin clauses). enumerate() would obscure the mathematical structure.
#![allow(clippy::needless_range_loop)]

//! RAIN Rust hot-kernels crate.

pub mod hymn;
pub mod nsga2;
pub mod sofar;
pub mod tsetlin;
pub mod vsa;

#[cfg(feature = "python")]
pub mod ffi_python;

#[cfg(feature = "wasm")]
pub mod ffi_wasm;

/// Crate version pinned at scaffold creation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the crate version (smoke-test export).
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
