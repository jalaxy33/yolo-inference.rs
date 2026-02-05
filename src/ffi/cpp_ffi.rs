//! C++ FFI module - exposes Rust functions to C++ via CXX

use cxx::CxxString;
use image::GenericImageView;
use std::path::PathBuf;
use ultralytics_inference as ul;

use crate::infer_fn::InferResult;
use crate::{Source, init_logger, parse_toml, run_online_prediction, run_prediction};

//================================================================================
// FFI Bridge - defines the C++ interface
//================================================================================

#[cxx::bridge]
mod ffi {
    /// Image metadata shared with C++
    struct ImageInfo {
        width: u32,
        height: u32,
        channels: u32,
    }

    extern "Rust" {
        // Types
        type RustImage;
        type InferResult;

        // Prediction functions
        fn predict_from_toml(config_toml: &CxxString);

        fn online_predict_from_toml(
            images: Vec<Box<RustImage>>,
            config_toml: &CxxString,
        ) -> Vec<Box<InferResult>>;

        // Image functions
        unsafe fn image_from_pixels(
            pixels: *const u8,
            width: u32,
            height: u32,
            channels: u32,
        ) -> Box<RustImage>;

        fn image_info(image: &RustImage) -> ImageInfo;
    }
}

use ffi::ImageInfo;

//================================================================================
// Internal Types
//================================================================================

/// Wrapper for image data (Opaque Type for C++)
pub struct RustImage {
    inner: image::DynamicImage,
}

impl RustImage {
    fn new(inner: image::DynamicImage) -> Self {
        Self { inner }
    }
}

//================================================================================
// Internal Functions
//================================================================================

/// Create RustImage from raw pixel buffer
///
/// # Safety
/// The `pixels` pointer must be valid and contain at least `width * height * channels` bytes
unsafe fn image_from_pixels(
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

fn image_info(image: &RustImage) -> ImageInfo {
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

//================================================================================
// Public API - functions exposed to C++
//================================================================================

/// Run prediction from TOML config file
fn predict_from_toml(config_toml: &CxxString) {
    init_logger();
    let args =
        parse_toml(&PathBuf::from(config_toml.to_string())).expect("Failed to parse TOML config");
    run_prediction(&args).expect("Prediction failed");
}

/// Online prediction with RustImage wrappers and TOML config
fn online_predict_from_toml(
    images: Vec<Box<RustImage>>,
    config_toml: &CxxString,
) -> Vec<Box<InferResult>> {
    init_logger();

    // Parse config
    let args =
        parse_toml(&PathBuf::from(config_toml.to_string())).expect("Failed to parse TOML config");

    // Load model
    let config: ul::InferenceConfig = (&args).try_into().expect("Invalid inference config");
    let mut model =
        ul::YOLOModel::load_with_config(&args.model, config).expect("Failed to load model");

    // Extract DynamicImage from wrappers
    let dynamic_images: Vec<image::DynamicImage> =
        images.into_iter().map(|wrapper| wrapper.inner).collect();

    // Create source and run inference
    let source = Source::ImageVec(dynamic_images);
    let results = run_online_prediction(&mut model, &source, &args).expect("Prediction failed");

    // Return Box<InferResult> list
    results
        .unwrap_or_default()
        .into_iter()
        .map(Box::new)
        .collect()
}
