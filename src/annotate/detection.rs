use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::RgbImage;
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ultralytics_inference as ul;

use super::AnnotateConfigs;
use super::annotate_uitls::rect_intersect;
use super::color::{get_class_color, get_text_color};

/// Draw object detection results (boxes and masks)
pub fn draw_detection(
    img: &mut RgbImage,
    result: &ul::Results,
    configs: &AnnotateConfigs,
    font: Option<&FontRef>,
) {
    draw_masks(img, result);
    draw_boxes_and_labels(img, result, configs, font);
}

fn draw_masks(img: &mut RgbImage, result: &ul::Results) {
    // Get boxes
    let boxes = match result.boxes.as_ref() {
        Some(b) => b,
        None => return, // No boxes to draw masks for
    };

    let (width, height) = img.dimensions();
    let xyxy = boxes.xyxy();
    let cls = boxes.cls();

    // Create an overlay image for masks to handle overlaps correctly
    let mut overlay = img.clone();
    let mut mask_present = false;

    // Draw masks onto the overlay
    if let Some(masks) = result.masks.as_ref() {
        let (mask_n, _mask_h, _mask_w) = masks.data.dim();

        for i in 0..boxes.len() {
            if i >= mask_n {
                break;
            }

            let class_id = cls[i] as usize;
            let color = get_class_color(class_id);
            let (r, g, b) = (color.0[0], color.0[1], color.0[2]);

            mask_present = true;

            let x1 = xyxy[[i, 0]].max(0.0).min(width as f32) as u32;
            let y1 = xyxy[[i, 1]].max(0.0).min(height as f32) as u32;
            let x2 = xyxy[[i, 2]].max(0.0).min(width as f32) as u32;
            let y2 = xyxy[[i, 3]].max(0.0).min(height as f32) as u32;

            for y in y1..y2 {
                for x in x1..x2 {
                    if masks.data[[i, y as usize, x as usize]] > 0.5 {
                        let pixel = overlay.get_pixel_mut(x, y);
                        pixel.0[0] = r;
                        pixel.0[1] = g;
                        pixel.0[2] = b;
                    }
                }
            }
        }
    }

    // Blend overlay with original image
    if mask_present {
        let alpha = 0.3;
        for y in 0..height {
            for x in 0..width {
                let p_img = img.get_pixel_mut(x, y);
                let p_overlay = overlay.get_pixel(x, y);

                p_img.0[0] = f32::from(p_overlay.0[0])
                    .mul_add(alpha, f32::from(p_img.0[0]) * (1.0 - alpha))
                    as u8;
                p_img.0[1] = f32::from(p_overlay.0[1])
                    .mul_add(alpha, f32::from(p_img.0[1]) * (1.0 - alpha))
                    as u8;
                p_img.0[2] = f32::from(p_overlay.0[2])
                    .mul_add(alpha, f32::from(p_img.0[2]) * (1.0 - alpha))
                    as u8;
            }
        }
    }
}

fn draw_boxes_and_labels(
    img: &mut RgbImage,
    result: &ul::Results,
    configs: &AnnotateConfigs,
    font: Option<&FontRef>,
) {
    let show_box = configs.show_box;
    let show_label = configs.show_label && show_box;
    let show_conf = configs.show_conf && show_label;

    if !show_box {
        return;
    }

    // Get boxes
    let boxes = match result.boxes.as_ref() {
        Some(b) => b,
        None => return, // No boxes to draw
    };

    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and font size
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let font_scale = (11.0 * scale_factor).max(10.0); // Min font size 10

    // Get box data
    let xyxy = boxes.xyxy();
    let conf = boxes.conf();
    let cls = boxes.cls();

    // Keep track of occupied label areas to avoid overlap
    let mut labels_rects: Vec<Rect> = Vec::new();

    // Draw boxes and labels
    for i in 0..boxes.len() {
        let class_id = cls[i] as usize;
        let confidence = conf[i];

        let mut x1 = xyxy[[i, 0]].round() as i32;
        let mut y1 = xyxy[[i, 1]].round() as i32;
        let mut x2 = xyxy[[i, 2]].round() as i32;
        let mut y2 = xyxy[[i, 3]].round() as i32;

        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
        }
        if y1 > y2 {
            std::mem::swap(&mut y1, &mut y2);
        }

        x1 = x1.max(0).min(width as i32 - 1);
        y1 = y1.max(0).min(height as i32 - 1);
        x2 = x2.max(0).min(width as i32 - 1);
        y2 = y2.max(0).min(height as i32 - 1);

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let color = get_class_color(class_id);

        // Draw box
        for t in 0..thickness {
            let tx1 = (x1 + t).min(x2);
            let ty1 = (y1 + t).min(y2);
            let tx2 = (x2 - t).max(tx1);
            let ty2 = (y2 - t).max(ty1);
            if tx2 > tx1 && ty2 > ty1 {
                let rect = Rect::at(tx1, ty1).of_size((tx2 - tx1) as u32, (ty2 - ty1) as u32);
                draw_hollow_rect_mut(img, rect, color);
            }
        }

        // Draw label and confidence
        if !show_label {
            continue;
        }

        let class_name = result.names.get(&class_id).map_or("object", String::as_str);
        let label = if show_conf {
            format!("{} {:.2}", class_name, confidence)
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
            // Default: above the box
            let mut text_x = x1;
            let mut text_y = y1 - text_h;

            // If label is out of image (top), move inside
            if text_y < 0 {
                text_y = y1;
            }

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
                    // Reached bottom, maybe try moving right?
                    // For now just stop here or loop around?
                    // Let's try moving x a bit and resetting y
                    text_y = y1 - text_h;
                    if text_y < 0 {
                        text_y = y1;
                    }
                    text_x += 10; // Shift right slightly
                    if text_x + text_w >= width as i32 {
                        // Screen full, just give up and draw wherever
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
