use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;
use ultralytics_inference as ul;

use crate::annotate::AnnotateConfigs;
use crate::error::{AppError, Result};
use crate::infer_fn::{InferFn, InferResult, auto_infer, deserialize_infer_fn};
use crate::source::{Source, deserialize_source};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PredictArgs {
    /// Path to ONNX model file
    pub model: PathBuf,

    /// Input source (image or directory)
    #[serde(default, deserialize_with = "deserialize_source")]
    pub source: Source,

    /// Confidence threshold
    pub conf: f32,

    /// IoU threshold for NMS
    pub iou: f32,

    /// Maximum number of detections
    pub max_det: usize,

    /// Inference image size
    pub imgsz: Option<usize>,

    /// Use FP16 half-precision inference
    pub half: bool,

    /// Batch size for inference
    pub batch: Option<usize>,

    /// Device to use (cpu, cuda:0, mps, coreml, directml:0, openvino, tensorrt:0, etc.)
    pub device: Option<String>,

    /// Directory to save results
    pub save_dir: Option<PathBuf>,

    /// Inference function to use
    #[serde(default, deserialize_with = "deserialize_infer_fn")]
    pub infer_fn: InferFn,

    /// Whether to generate annotations
    pub annotate: bool,

    /// Annotate configurations
    pub annotate_cfg: AnnotateConfigs,

    /// Multi-thread channel capacity
    pub channel_capacity: Option<usize>,

    /// Whether to store and return inference results
    pub return_result: bool,

    /// Show verbose output
    pub verbose: bool,
}

impl Default for PredictArgs {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            source: Default::default(),
            conf: 0.25,
            iou: 0.45,
            max_det: 300,
            imgsz: None,
            half: false,
            batch: Some(4),
            device: None,
            save_dir: None,
            infer_fn: Default::default(),
            annotate: false,
            annotate_cfg: Default::default(),
            channel_capacity: Some(8),
            return_result: false,
            verbose: false,
        }
    }
}

impl TryFrom<&PredictArgs> for ul::InferenceConfig {
    type Error = AppError;

    fn try_from(args: &PredictArgs) -> std::result::Result<Self, Self::Error> {
        let mut config = Self::new()
            .with_confidence(args.conf)
            .with_iou(args.iou)
            .with_half(args.half)
            .with_max_det(args.max_det)
            .with_batch(args.batch.unwrap_or(1));

        if let Some(sz) = args.imgsz {
            config = config.with_imgsz(sz, sz);
        }

        if let Some(ref device_str) = args.device {
            let device: ul::Device = device_str
                .parse()
                .map_err(|_| AppError::InvalidDevice(device_str.clone()))?;
            config = config.with_device(device);
        }

        Ok(config)
    }
}

/// Core prediction API
///
/// Returns:
/// - Optionally, a Vec of ('annotated image', 'source meta') if `return_annotated` is true
pub fn run_prediction(args: &PredictArgs) -> Result<Option<Vec<InferResult>>> {
    let start_time = Instant::now();

    // Load model using TryFrom trait
    let config: ul::InferenceConfig = args.try_into()?;
    let mut model = ul::YOLOModel::load_with_config(&args.model, config)
        .map_err(|e| AppError::ModelLoad(e.to_string()))?;

    // Select infer_fn: Sequential for single image, user choice for batch
    let infer_fn = if args.source.is_image() {
        InferFn::Sequential
    } else {
        args.infer_fn.clone()
    };

    // Perform inference
    let mut final_results = if args.return_result {
        Some(Vec::new())
    } else {
        None
    };

    auto_infer(
        &mut model,
        &args.source,
        &infer_fn,
        args,
        &mut final_results,
    )?;

    // Log total duration
    let duration = start_time.elapsed();
    tracing::info!("Total prediction time: {:.3?}", duration);

    Ok(final_results)
}

/// Online prediction - reuses an existing model for inference.
/// Only supports `Source::Image` and `Source::ImageVec`.
/// Uses `Sequential` for single image, `args.infer_fn` for batch.
pub fn run_online_prediction(
    model: &mut ul::YOLOModel,
    source: &Source,
    args: &PredictArgs,
) -> Result<Option<Vec<InferResult>>> {
    let start_time = Instant::now();

    // Only accept in-memory images
    if !matches!(source, Source::Image(_) | Source::ImageVec(_)) {
        return Err(AppError::Config(
            "Online prediction only supports Source::Image or Source::ImageVec".to_string(),
        ));
    }

    // Select infer_fn: Sequential for single image, user choice for batch
    let infer_fn = if source.is_image() {
        InferFn::Sequential
    } else {
        args.infer_fn.clone()
    };

    // Perform inference
    let mut final_results = if args.return_result {
        Some(Vec::new())
    } else {
        None
    };

    auto_infer(model, source, &infer_fn, args, &mut final_results)?;

    // Log total duration
    let duration = start_time.elapsed();
    tracing::info!("Total prediction time: {:.3?}", duration);

    Ok(final_results)
}
