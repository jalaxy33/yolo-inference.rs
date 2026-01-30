// -- submodules
mod batch_channel_ppl;
mod batch_sequential;
mod batch_utils;
mod channel_ppl;
mod sequential;

pub use batch_channel_ppl::batch_channel_pipeline_infer;
pub use batch_sequential::batch_sequential_infer;
pub use channel_ppl::channel_pipeline_infer;
pub use sequential::sequential_infer;

// -- external imports
use image::DynamicImage;
use serde::Deserialize;
use std::str::FromStr;
use strum::{Display, EnumString, VariantNames};
use ultralytics_inference as ul;

use crate::error::Result;
use crate::predict::PredictArgs;
use crate::source::SourceMeta;

// -- enums

#[derive(Debug, Clone, EnumString, Display, Deserialize, VariantNames)]
#[serde(untagged)]
/// Inference function to use
pub enum InferFn {
    #[strum(serialize = "Sequential")]
    Sequential,

    #[strum(serialize = "BatchSequential")]
    BatchSequential,

    #[strum(serialize = "ChannelPipeline")]
    ChannelPipeline,

    #[strum(serialize = "BatchChannelPipeline")]
    BatchChannelPipeline,
}

impl Default for InferFn {
    fn default() -> Self {
        InferFn::BatchChannelPipeline
    }
}

/// Custom deserializer with helpful error message
pub fn deserialize_infer_fn<'de, D>(deserializer: D) -> Result<InferFn, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    InferFn::from_str(&value).map_err(|_| {
        let variants = InferFn::VARIANTS;
        serde::de::Error::invalid_value(
            serde::de::Unexpected::Str(&value),
            &format!("one of {}", variants.join(", ")).as_str(),
        )
    })
}

// -- structs

/// Inference result for a single image/frame
#[derive(Debug)]
pub struct InferResult {
    /// Raw inference results from yolo model
    pub result: ul::Results,

    /// Annotated image after inference
    pub annotated: Option<DynamicImage>,

    /// Source meta information
    pub meta: SourceMeta,
}

// -- public API

/// Perform model inference.
///
/// - You can choose inference functions via `infer_fn`.
/// - To avoid huge memory consumption, results are returned via `return_results` argument if
///   provided.
pub fn auto_infer(
    model: &mut ul::YOLOModel,
    infer_fn: &InferFn,
    args: &PredictArgs,
    return_results: &mut Option<Vec<InferResult>>,
) -> Result<()> {
    match infer_fn {
        InferFn::Sequential => sequential_infer(model, args, return_results)?,
        InferFn::BatchSequential => batch_sequential_infer(model, args, return_results)?,
        InferFn::ChannelPipeline => channel_pipeline_infer(model, args, return_results)?,
        InferFn::BatchChannelPipeline => batch_channel_pipeline_infer(model, args, return_results)?,
    }
    Ok(())
}
