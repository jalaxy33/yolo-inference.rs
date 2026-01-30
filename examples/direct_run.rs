/// An example of directly running prediction without using config files.
use std::path::PathBuf;

use anyhow::{Context, Result};
use yolo_inference::{AnnotateConfigs, InferFn, PredictArgs, Source, init_logger, run_prediction};

#[allow(dead_code)]
enum DataScale {
    Image,
    SmallBatch,
    LargeBatch,
}

#[allow(dead_code)]
enum Checkpoint {
    Unbatchable,
    Batchable,
}

fn main() -> Result<()> {
    init_logger();

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let checkpoint_dir = project_root.join("assets/checkpoints");
    let save_dir = project_root.join("results/demo");

    let chekpoint_type = Checkpoint::Batchable;
    let model_path = match chekpoint_type {
        Checkpoint::Unbatchable => checkpoint_dir.join("unbatch-seg.onnx"),
        Checkpoint::Batchable => checkpoint_dir.join("yolo11n-seg.onnx"),
    };

    let data_scale = DataScale::SmallBatch;
    let source_path = match data_scale {
        DataScale::Image => project_root.join("assets/images/small-batch/bus.jpg"),
        DataScale::SmallBatch => project_root.join("assets/images/small-batch"),
        DataScale::LargeBatch => project_root.join("assets/images/coco128"),
    };

    let annotate_cfg = AnnotateConfigs {
        on_blank: false,
        show_box: true,
        show_label: true,
        show_conf: true,
        ..Default::default()
    };

    let args = PredictArgs {
        model: model_path,
        source: Source::from(source_path),
        half: true,
        batch: Some(8),
        device: Some("cuda".to_string()),
        save_dir: Some(save_dir),
        infer_fn: InferFn::BatchChannelPipeline,
        annotate: true,
        annotate_cfg,
        return_result: false,
        verbose: false,
        ..Default::default()
    };

    if let Some(final_result) =
        run_prediction(&args).with_context(|| "Failed to run prediction".to_string())?
    {
        tracing::info!("Total annotated images returned: {}", final_result.len());
    }

    Ok(())
}
