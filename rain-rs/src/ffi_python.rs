// CONFIDENTIAL - PATENT PENDING
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

//! PyO3 bindings for the rain-rs hot kernels.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::hymn::{HymnConfig, HymnModel};

#[pyclass]
pub struct PyHymnModel {
    inner: HymnModel,
}

#[pymethods]
impl PyHymnModel {
    #[new]
    #[pyo3(signature = (in_dim=10000, hidden_dim=4096, out_dim=10000, dropout=0.0, seed=42))]
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        dropout: f32,
        seed: u64,
    ) -> Self {
        let config = HymnConfig {
            in_dim,
            hidden_dim,
            out_dim,
            dropout,
            seed,
        };
        PyHymnModel {
            inner: HymnModel::new(config),
        }
    }

    fn weights_w1<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let cfg = self.inner.config();
        let data = self.inner.weights_w1().to_vec();
        let arr = numpy::ndarray::Array2::from_shape_vec((cfg.in_dim, cfg.hidden_dim), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    fn weights_w2<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let cfg = self.inner.config();
        let data = self.inner.weights_w2().to_vec();
        let arr = numpy::ndarray::Array2::from_shape_vec((cfg.hidden_dim, cfg.out_dim), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, i16>,
        input: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let state_slice = state.as_slice()?;
        let input_slice = input.as_slice()?;
        let out = self.inner.forward(state_slice, input_slice);
        Ok(out.into_pyarray(py))
    }
}

/// Module name must match `tool.maturin.module-name` leaf: `_rust`.
#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHymnModel>()?;
    Ok(())
}
