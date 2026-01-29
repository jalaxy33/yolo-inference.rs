mod annotate;
mod ffi;
mod infer_fn;
mod logging;
mod predict;
mod progress_bar;
mod source;
mod toml_utils;

pub use annotate::{AnnotateConfigs, annotate_image};
pub use infer_fn::{InferFn, auto_infer};
pub use logging::init_logger;
pub use progress_bar::progress_bar_style;
pub use source::{BatchSourceLoader, Source, SourceLoader, SourceMeta};
pub use toml_utils::parse_toml;

// Core inference function
pub use predict::{PredictArgs, run_prediction};

// FFI
#[allow(unused_imports)]
pub use ffi::*;
