mod annotate;
mod error;
mod ffi;
mod infer_fn;
mod logging;
mod predict;
mod progress_bar;
mod source;
mod toml_utils;

pub use annotate::{AnnotateConfigs, annotate_image};
pub use error::{AppError, Result};
pub use infer_fn::{InferFn, auto_infer};
pub use logging::init_logger;
pub use progress_bar::progress_bar_style;
pub use source::{BatchSourceLoader, Source, SourceLoader, SourceMeta, collect_images_from_dir,
                 is_image_file};
pub use toml_utils::parse_toml;

// Core inference function
pub use predict::{PredictArgs, run_online_prediction, run_prediction};

// FFI
#[allow(unused_imports)]
pub use ffi::*;
