use cxx::CxxString;
use std::path::PathBuf;

use crate::{init_logger, parse_toml, run_prediction};

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        // -- Functions exposed to C++
        fn predict_from_toml(config_toml: &CxxString);
    }
}

// -- Functions exposed to C++

fn predict_from_toml(config_toml: &CxxString) {
    init_logger();
    let args =
        parse_toml(&PathBuf::from(config_toml.to_string())).expect("Failed to parse TOML config");
    run_prediction(&args).expect("Prediction failed");
}
