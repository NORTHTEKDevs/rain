// CONFIDENTIAL
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

//! WASM FFI bindings for rain-rs hot kernels.
//!
//! Stub scaffold. Full WASM bindings land when browser inference is needed.
//! Build with `--features wasm`.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

/// Returns the crate version. Smoke-test export to verify WASM build pipeline.
#[wasm_bindgen]
pub fn version() -> String {
    crate::VERSION.to_string()
}
