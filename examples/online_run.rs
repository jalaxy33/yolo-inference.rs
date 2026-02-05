/// Example of running online prediction without recreating the model.
use std::path::PathBuf;

use anyhow::{Context, Result};
use ultralytics_inference as ul;
use yolo_inference::{collect_images_from_dir, init_logger, parse_toml, run_online_prediction};

fn main() -> Result<()> {
    init_logger();

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config_dir = project_root.join("assets/configs/");
    let config_toml = config_dir.join("online-predict.toml");

    // Parse config (without source field)
    let args = parse_toml(&config_toml)
        .with_context(|| format!("Failed to parse TOML config: {:?}", config_toml))?;

    tracing::info!("Config loaded, source: {:?}", args.source);

    // Load model once using TryFrom trait
    let config: ul::InferenceConfig = (&args).try_into()?;
    let mut model = ul::YOLOModel::load_with_config(&args.model, config)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    tracing::info!("Model loaded successfully");
    tracing::info!("Using infer_fn: {:?}", args.infer_fn);

    // Read images from small-batch directory
    let image_dir = project_root.join("assets/images/small-batch/");
    let image_paths = collect_images_from_dir(&image_dir)?;

    tracing::info!("Found {} images in {:?}", image_paths.len(), image_dir);

    // Process each image as online prediction
    for (idx, path) in image_paths.iter().enumerate() {
        let image = image::open(path)?;
        let source = yolo_inference::Source::Image(image);

        let results = run_online_prediction(&mut model, source, &args)?;

        if let Some(ref res) = results {
            tracing::info!("Image {}: processed {} results", idx, res.len());
        }
    }

    tracing::info!("Online prediction example completed");
    Ok(())
}
