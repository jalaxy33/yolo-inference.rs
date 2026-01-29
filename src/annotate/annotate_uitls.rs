use image::{Rgb, RgbImage};
use imageproc::rect::Rect;

pub fn rect_intersect(r1: &Rect, r2: &Rect) -> bool {
    let r1_left = r1.left();
    let r1_right = r1.right();
    let r1_top = r1.top();
    let r1_bottom = r1.bottom();

    let r2_left = r2.left();
    let r2_right = r2.right();
    let r2_top = r2.top();
    let r2_bottom = r2.bottom();

    !(r2_left >= r1_right || r2_right <= r1_left || r2_top >= r1_bottom || r2_bottom <= r1_top)
}

/// Draw a line segment on an image
pub fn draw_line_segment(
    img: &mut RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgb<u8>,
    thickness: i32,
) {
    let (width, height) = img.dimensions();

    // Bresenham's line algorithm with thickness
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1.0 } else { -1.0 };
    let sy = if y1 < y2 { 1.0 } else { -1.0 };
    let mut err = dx - dy;

    let mut x = x1;
    let mut y = y1;

    loop {
        // Draw a thick point
        let half_t = thickness / 2;
        for tx in -half_t..=half_t {
            for ty in -half_t..=half_t {
                let px = (x as i32 + tx).max(0).min(width as i32 - 1) as u32;
                let py = (y as i32 + ty).max(0).min(height as i32 - 1) as u32;
                img.put_pixel(px, py, color);
            }
        }

        if (x - x2).abs() < 1.0 && (y - y2).abs() < 1.0 {
            break;
        }

        let e2 = 2.0 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a filled circle on an image
pub fn draw_filled_circle(img: &mut RgbImage, cx: i32, cy: i32, radius: i32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius * radius
                && x >= 0
                && y >= 0
                && x < width as i32
                && y < height as i32
            {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

/// Draw a transparent rectangle on an image
pub fn draw_transparent_rect(
    img: &mut RgbImage,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    color: Rgb<u8>,
    alpha: f32,
) {
    let (width, height) = img.dimensions();
    let alpha = alpha.max(0.0).min(1.0);
    let inv_alpha = 1.0 - alpha;

    let r = f32::from(color[0]);
    let g = f32::from(color[1]);
    let b = f32::from(color[2]);

    for dy in 0..h {
        let py = y + dy as i32;
        if py < 0 || py >= height as i32 {
            continue;
        }

        for dx in 0..w {
            let px = x + dx as i32;
            if px < 0 || px >= width as i32 {
                continue;
            }

            let pixel = img.get_pixel_mut(px as u32, py as u32);
            let current = pixel.0;

            let new_r = f32::from(current[0]).mul_add(inv_alpha, r * alpha) as u8;
            let new_g = f32::from(current[1]).mul_add(inv_alpha, g * alpha) as u8;
            let new_b = f32::from(current[2]).mul_add(inv_alpha, b * alpha) as u8;

            *pixel = Rgb([new_r, new_g, new_b]);
        }
    }
}
