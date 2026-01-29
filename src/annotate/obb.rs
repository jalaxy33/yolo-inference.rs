use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::RgbImage;
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ultralytics_inference as ul;

use super::AnnotateConfigs;
use super::annotate_uitls::{draw_line_segment, rect_intersect};
use super::color::{get_class_color, get_text_color};

/// Draw oriented bounding boxes (OBB)
pub fn draw_obb(
    img: &mut RgbImage,
    result: &ul::Results,
    configs: &AnnotateConfigs,
    font: Option<&FontRef>,
) {
    let show_conf = configs.show_conf;

    let obb = match &result.obb {
        Some(o) => o,
        None => return,
    };

    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and font size
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let font_scale = (11.0 * scale_factor).max(10.0); // Min font size 10

    let corners = obb.xyxyxyxy();
    let conf = obb.conf();
    let cls = obb.cls();

    // Keep track of occupied label areas to avoid overlap
    let mut labels_rects: Vec<Rect> = Vec::new();

    for i in 0..obb.len() {
        let class_id = cls[i] as usize;
        let color = get_class_color(class_id);

        for j in 0..4 {
            let next_j = (j + 1) % 4;
            let x1 = corners[[i, j, 0]];
            let y1 = corners[[i, j, 1]];
            let x2 = corners[[i, next_j, 0]];
            let y2 = corners[[i, next_j, 1]];
            draw_line_segment(img, x1, y1, x2, y2, color, thickness);
        }

        let class_name = result.names.get(&class_id).map_or("object", String::as_str);
        let label = if show_conf {
            format!("{} {:.2}", class_name, conf[i])
        } else {
            class_name.to_string()
        };

        if let Some(f) = font {
            let scale = PxScale::from(font_scale);
            let scaled_font = f.as_scaled(scale);
            let mut text_w = 0.0;
            for c in label.chars() {
                text_w += scaled_font.h_advance(scaled_font.glyph_id(c));
            }
            let text_w = text_w.ceil() as i32;
            let text_h = scale.y.ceil() as i32;

            // Smart label placement
            // Default: at the first corner (usually top-left-ish)
            let mut text_x = corners[[i, 0, 0]] as i32;
            let mut text_y = (corners[[i, 0, 1]] as i32 - text_h).max(0);

            // If label is out of image (left), move right
            if text_x < 0 {
                text_x = 0;
            }

            // If label is out of image (right), move left
            if text_x + text_w >= width as i32 {
                text_x = width as i32 - text_w - 1;
            }

            // Check bounds one last time (bottom)
            if text_y + text_h >= height as i32 {
                text_y = height as i32 - text_h - 1;
            }

            // Overlap avoidance
            // Check against existing labels. If overlap, move down.
            let mut attempts = 0;
            let max_attempts = 10;
            let mut current_rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);

            'placement: while attempts < max_attempts {
                let mut is_overlapping = false;
                for existing in &labels_rects {
                    // Simple intersection check
                    if rect_intersect(&current_rect, existing) {
                        is_overlapping = true;
                        break;
                    }
                }

                if !is_overlapping {
                    break 'placement;
                }

                // Move down
                text_y += text_h; // stack below

                // Check bounds again
                if text_y + text_h >= height as i32 {
                    // Reached bottom, try resetting y and moving x
                    text_y = (corners[[i, 0, 1]] as i32 - text_h).max(0);
                    text_x += 10; // Shift right
                    if text_x + text_w >= width as i32 {
                        break 'placement;
                    }
                }

                current_rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);
                attempts += 1;
            }

            // Add to occupied list
            labels_rects.push(current_rect);

            if text_x >= 0
                && text_y >= 0
                && text_x + text_w < width as i32
                && text_y + text_h < height as i32
            {
                draw_filled_rect_mut(img, current_rect, color);
                let text_color = get_text_color(color);
                draw_text_mut(img, text_color, text_x, text_y, scale, f, &label);
            }
        }
    }
}
