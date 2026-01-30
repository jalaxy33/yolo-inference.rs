// -- submodules
mod annotate_uitls;
mod classification;
mod color;
mod detection;
mod font;
mod obb;
mod pose;

use classification::draw_classification;
use detection::draw_detection;
use font::{is_ascii, load_font};
use obb::draw_obb;
use pose::draw_pose;
use serde::Deserialize;

// -- external imports
use crate::error::Result;
use ab_glyph::FontRef;
use image::{DynamicImage, GenericImageView, RgbImage};
use ultralytics_inference as ul;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AnnotateConfigs {
    /// whether to draw on a blank image
    pub on_blank: bool,

    /// whether to show boxes
    pub show_box: bool,

    /// whether to show class labels
    pub show_label: bool,

    /// whether to show confidence scores
    pub show_conf: bool,

    /// (top-k) number of classification results to show
    pub top_k: Option<usize>,
}

impl Default for AnnotateConfigs {
    fn default() -> Self {
        Self {
            on_blank: false,
            show_box: true,
            show_label: true,
            show_conf: true,
            top_k: Some(5),
        }
    }
}

pub fn annotate_image(
    img: &DynamicImage,
    result: &ul::Results,
    configs: &AnnotateConfigs,
) -> Result<DynamicImage> {
    let on_blank = configs.on_blank;
    let show_box = configs.show_box;
    let show_label = configs.show_label && show_box;
    let top_k = configs.top_k;

    let have_obb = result.obb.is_some();
    let have_probs = result.probs.is_some();

    // Prepare result image
    let mut annotated = if on_blank {
        let (w, h) = img.dimensions();
        RgbImage::new(w, h)
    } else {
        img.to_rgb8()
    };

    // Prepare font if needed
    let font_data = if show_label || have_obb || have_probs {
        let mut use_unicode_font = false;
        if result.boxes.is_some() {
            for name in result.names.values() {
                if !is_ascii(name) {
                    use_unicode_font = true;
                    break;
                }
            }
        }
        let font_name = if use_unicode_font {
            "Arial.Unicode.ttf"
        } else {
            "Arial.ttf"
        };

        let font_data = load_font(font_name);
        font_data
    } else {
        None
    };

    let font = match font_data {
        Some(ref data) => FontRef::try_from_slice(data).ok(),
        None => None,
    };

    // Draw annotations
    draw_detection(&mut annotated, result, configs, font.as_ref());
    draw_pose(&mut annotated, result, None, None, None);
    draw_obb(&mut annotated, result, configs, font.as_ref());
    draw_classification(&mut annotated, result, font.as_ref(), top_k.unwrap_or(5));

    Ok(DynamicImage::ImageRgb8(annotated))
}
