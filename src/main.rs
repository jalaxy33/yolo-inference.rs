/// An example of running different experiments based on TOML config files.
/// Copied from `examples/toml_run.rs`
use std::path::PathBuf;

use yolo_inference::{init_logger, parse_toml, run_prediction};

#[allow(dead_code)]
enum Experiment {
    OneImage,
    SmallBatch,
    LargeBatch,
    UnbatchableModel,
}

fn main() {
    init_logger();

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config_dir = project_root.join("assets/configs/");

    let experiment = Experiment::SmallBatch;
    let config_toml = match experiment {
        Experiment::OneImage => config_dir.join("one-image.toml"),
        Experiment::SmallBatch => config_dir.join("small-batch.toml"),
        Experiment::LargeBatch => config_dir.join("large-batch.toml"),
        Experiment::UnbatchableModel => config_dir.join("unbatchable-model.toml"),
    };

    let args = parse_toml(&config_toml).expect("Failed to parse TOML config");

    dbg!(&args);

    if let Some(final_result) = run_prediction(&args).expect("Failed to run prediction") {
        tracing::info!("Total annotated images returned: {}", final_result.len());
    }
}
