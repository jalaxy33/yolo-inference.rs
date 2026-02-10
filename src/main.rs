/// An example of running different experiments based on TOML config files.
/// Copied from `examples/toml_run.rs`
use std::path::PathBuf;

use anyhow::{Context, Result};
use yolo_inference::{init_logger, parse_toml, run_prediction};

#[allow(dead_code)]
enum Experiment {
    OneImage,
    SmallBatch,
    LargeBatch,
    UnbatchableModel,
}

fn main() -> Result<()> {
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

    let args = parse_toml(&config_toml, &project_root)
        .with_context(|| format!("Failed to parse TOML config: {:?}", config_toml))?;

    dbg!(&args);

    if let Some(final_result) = run_prediction(&args)
        .with_context(|| format!("Failed to run prediction: {:?}", config_toml))?
    {
        tracing::info!("Total annotated images returned: {}", final_result.len());
    }

    Ok(())
}
