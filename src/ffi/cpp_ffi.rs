//! C++ FFI module - exposes Rust functions to C++ via CXX

use cxx::CxxString;
use image::GenericImageView;
use std::path::PathBuf;
use ultralytics_inference as ul;

use crate::infer_fn::InferResult;
use crate::{Source, init_logger, parse_toml, run_online_prediction, run_prediction};

//================================================================================
// FFI Bridge
//================================================================================

#[cxx::bridge]
mod ffi {
    pub struct ImageInfo {
        width: u32,
        height: u32,
        channels: u32,
    }

    extern "Rust" {
        type RustImage;
        type InferResult;

        // Image operations
        unsafe fn image_from_pixels(
            pixels: *const u8,
            width: u32,
            height: u32,
            channels: u32,
        ) -> Box<RustImage>;
        fn image_info(image: &RustImage) -> ImageInfo;
        fn is_image_empty(image: &RustImage) -> bool;

        // Prediction operations
        fn predict_from_toml(config_toml: &CxxString);
        fn online_predict_from_toml(
            images: Vec<Box<RustImage>>,
            config_toml: &CxxString,
        ) -> Vec<Box<InferResult>>;

        // InferResult accessors
        fn get_result_annotated(result: &InferResult) -> Box<RustImage>;
        fn take_result_annotated(result: &mut InferResult) -> Box<RustImage>;
    }
}

use ffi::ImageInfo;

//================================================================================
// Types
//================================================================================

pub struct RustImage {
    pub inner: image::DynamicImage,
}

impl RustImage {
    pub fn new(inner: image::DynamicImage) -> Self {
        Self { inner }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.width() == 0 || self.inner.height() == 0
    }
}

//================================================================================
// Image Operations
//================================================================================

/// Create a RustImage from raw pixel buffer.
/// `pixels` must be valid for `width * height * channels` bytes.
/// Supports 1 (grayscale), 3 (RGB), or 4 (RGBA) channels.
pub unsafe fn image_from_pixels(
    pixels: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> Box<RustImage> {
    assert!(!pixels.is_null(), "Pixel pointer is null");
    assert!(width > 0 && height > 0, "Invalid image dimensions");

    let data_len = (width * height * channels) as usize;
    let pixel_data = unsafe { std::slice::from_raw_parts(pixels, data_len) };

    let dynamic_image = match channels {
        1 => image::GrayImage::from_raw(width, height, pixel_data.to_vec())
            .map(image::DynamicImage::ImageLuma8)
            .expect("Failed to create grayscale image"),
        3 => image::RgbImage::from_raw(width, height, pixel_data.to_vec())
            .map(image::DynamicImage::ImageRgb8)
            .expect("Failed to create RGB image"),
        4 => image::RgbaImage::from_raw(width, height, pixel_data.to_vec())
            .map(image::DynamicImage::ImageRgba8)
            .expect("Failed to create RGBA image"),
        _ => panic!("Unsupported channel count: {}", channels),
    };

    Box::new(RustImage::new(dynamic_image))
}

/// Get image metadata (width, height, channels).
pub fn image_info(image: &RustImage) -> ImageInfo {
    let (w, h) = image.inner.dimensions();
    let channels = match &image.inner {
        image::DynamicImage::ImageLuma8(_) => 1,
        image::DynamicImage::ImageRgb8(_) => 3,
        image::DynamicImage::ImageRgba8(_) => 4,
        _ => 3,
    };
    ImageInfo {
        width: w,
        height: h,
        channels,
    }
}

/// Check if image is empty (width == 0 or height == 0).
/// Useful for checking if get_result_annotated returned valid data.
pub fn is_image_empty(image: &RustImage) -> bool {
    image.is_empty()
}

//================================================================================
// Prediction Operations
//================================================================================

/// Run batch prediction from TOML config file path.
pub fn predict_from_toml(config_toml: &CxxString) {
    init_logger();
    let args =
        parse_toml(&PathBuf::from(config_toml.to_string())).expect("Failed to parse TOML config");
    run_prediction(&args).expect("Prediction failed");
}

/// Run online prediction with in-memory images.
/// Takes images and TOML config, returns inference results.
pub fn online_predict_from_toml(
    images: Vec<Box<RustImage>>,
    config_toml: &CxxString,
) -> Vec<Box<InferResult>> {
    init_logger();

    let args =
        parse_toml(&PathBuf::from(config_toml.to_string())).expect("Failed to parse TOML config");

    let config: ul::InferenceConfig = (&args).try_into().expect("Invalid inference config");
    let mut model =
        ul::YOLOModel::load_with_config(&args.model, config).expect("Failed to load model");

    let dynamic_images: Vec<image::DynamicImage> =
        images.into_iter().map(|wrapper| wrapper.inner).collect();

    let source = Source::ImageVec(dynamic_images);
    let results = run_online_prediction(&mut model, &source, &args).expect("Prediction failed");

    results
        .unwrap_or_default()
        .into_iter()
        .map(Box::new)
        .collect()
}

//================================================================================
// InferResult Operations
//================================================================================

/// Clone the annotated image from InferResult.
/// Returns a copy of the annotated image. Original data is preserved in InferResult.
/// Returns empty image (0x0) if annotated is None. Use is_image_empty() to check.
pub fn get_result_annotated(result: &InferResult) -> Box<RustImage> {
    let img = result
        .annotated
        .as_ref()
        .cloned()
        .unwrap_or_else(|| image::DynamicImage::new_rgba8(0, 0));
    Box::new(RustImage::new(img))
}

/// Take ownership of the annotated image from InferResult.
/// Moves the annotated image out of InferResult (leaves None). More efficient than get_.
/// Returns empty image (0x0) if annotated is None. Use is_image_empty() to check.
pub fn take_result_annotated(result: &mut InferResult) -> Box<RustImage> {
    let img = result
        .annotated
        .take()
        .unwrap_or_else(|| image::DynamicImage::new_rgba8(0, 0));
    Box::new(RustImage::new(img))
}
