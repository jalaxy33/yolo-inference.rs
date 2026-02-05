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

/// Core prediction API
///
/// Returns:
/// - Optionally, a Vec of ('annotated image', 'source meta') if `return_annotated` is true
pub fn run_prediction(args: &PredictArgs) -> Result<Option<Vec<InferResult>>> {
    let start_time = Instant::now();

    // Parse arguments
    let model_path = &args.model;
    let conf_threshold = args.conf;
    let iou_threshold = args.iou;
    let max_detections = args.max_det;
    let img_size = args.imgsz;
    let use_half = args.half;
    let device: Option<ul::Device> = args
        .device
        .as_ref()
        .map(|d| d.parse().map_err(|_| AppError::InvalidDevice(d.clone())))
        .transpose()?;
    let batch_size = args.batch.unwrap_or(1);
    let infer_fn = &args.infer_fn;
    let return_annotated = args.return_result;

    // Load model with configurations
    let mut config = ul::InferenceConfig::new()
        .with_confidence(conf_threshold)
        .with_iou(iou_threshold)
        .with_half(use_half)
        .with_max_det(max_detections)
        .with_batch(batch_size);

    if let Some(sz) = img_size {
        config = config.with_imgsz(sz, sz);
    }
    if let Some(dev) = device {
        config = config.with_device(dev);
    }

    let mut model = ul::YOLOModel::load_with_config(model_path, config)
        .map_err(|e| AppError::ModelLoad(e.to_string()))?;

    // Perform inference
    let mut final_results = if return_annotated {
        Some(Vec::new())
    } else {
        None
    };
    auto_infer(&mut model, infer_fn, args, &mut final_results)?;

    // Log total duration
    let duration = start_time.elapsed();
    tracing::info!("Total prediction time: {:.3?}", duration);

    Ok(final_results)
}
