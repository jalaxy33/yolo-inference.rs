use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::*;

use crate::{init_logger, parse_toml, run_prediction};

#[pyfunction]
#[gen_stub_pyfunction]
pub fn predict_from_toml(config_toml: &str) {
    init_logger();
    let args =
        parse_toml(&std::path::PathBuf::from(config_toml)).expect("Failed to parse TOML config");
    run_prediction(&args).expect("Prediction failed");
}

/// Export rust library as Python module.
#[pymodule]
fn yolo_inference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_from_toml, m)?)?;
    Ok(())
}

// Generate stub info for this module.
define_stub_info_gatherer!(stub_info);
