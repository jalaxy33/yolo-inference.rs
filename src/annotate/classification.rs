use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use ultralytics_inference as ul;

use super::annotate_uitls::draw_transparent_rect;

/// Draw classification results
pub fn draw_classification(
    img: &mut RgbImage,
    result: &ul::Results,
    font: Option<&FontRef>,
    top_k: usize,
) {
    let probs = match &result.probs {
        Some(p) => p,
        None => return,
    };

    let font = match font {
        Some(f) => f,
        None => return,
    };

    let top_indices = probs.top_k(top_k);
    let (width, _height) = img.dimensions();

    // Adaptive font scale based on image width
    let scale_factor = (width as f32 / 600.0).max(0.6).min(2.0);
    let base_size = 30.0;
    let scale = PxScale::from(base_size * scale_factor);
    let line_height = (scale.y * 1.2) as i32;

    let x_pos = (20.0 * scale_factor) as i32;
    let mut y_pos = (20.0 * scale_factor) as i32;

    // Calculate max text width for background box
    let mut max_width = 0;
    let mut entries = Vec::new();

    for &class_id in &top_indices {
        let score = probs.data[class_id];
        if score < 0.01 {
            continue;
        }

        let class_name = result.names.get(&class_id).map_or("class", String::as_str);

        let label = format!("{class_name} {score:.2}");
        entries.push(label);
    }

    // Basic approximation: chars * scale * 0.5 (average char width)
    for label in &entries {
        let w = (label.len() as f32 * scale.x * 0.5) as u32;
        if w > max_width {
            max_width = w;
        }
    }

    // Draw background for all entries with padding
    let box_height = (entries.len() as i32 * line_height) + 10;
    let box_width = max_width + 20;

    if !entries.is_empty() {
        draw_transparent_rect(
            img,
            x_pos - 5,
            y_pos - 5,
            box_width,
            box_height as u32,
            Rgb([0, 0, 0]),
            0.4, // 40% opacity black tint
        );
    }

    for label in entries {
        // Draw text (white)
        draw_text_mut(img, Rgb([255, 255, 255]), x_pos, y_pos, scale, font, &label);

        y_pos += line_height;
    }
}
