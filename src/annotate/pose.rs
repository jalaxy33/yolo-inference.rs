use image::{Rgb, RgbImage};
use ultralytics_inference as ul;

use super::annotate_uitls::{draw_filled_circle, draw_line_segment};
use super::color::POSE_COLORS;

/// COCO-Pose dataset skeleton structure (pairs of keypoint indices)
/// Defines which keypoints connect to form the pose skeleton
pub const SKELETON: [[usize; 2]; 19] = [
    [15, 13], // right ankle to left knee
    [13, 11], // left knee to left hip
    [16, 14], // right ankle (16) to right knee
    [14, 12], // right knee to right hip
    [11, 12], // left hip to right hip
    [5, 11],  // left shoulder to left hip
    [6, 12],  // right shoulder to right hip
    [5, 6],   // left shoulder to right shoulder
    [5, 7],   // left shoulder to left elbow
    [6, 8],   // right shoulder to right elbow
    [7, 9],   // left elbow to left wrist
    [8, 10],  // right elbow to right wrist
    [1, 2],   // left eye to right eye
    [0, 1],   // nose to left eye
    [0, 2],   // nose to right eye
    [1, 3],   // left eye to left ear
    [2, 4],   // right eye to right ear
    [3, 5],   // left ear to left shoulder
    [4, 6],   // right ear to right shoulder
];

/// Limb color indices mapping to `POSE_COLORS`
/// Defines which color from the pose palette to use for each limb
/// Mapping: arms=blue, legs=orange, face=green
pub const LIMB_COLOR_INDICES: [usize; 19] = [
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16,
];

/// Keypoint color indices mapping to `POSE_COLORS`
/// Defines which color from the pose palette to use for each keypoint
/// Mapping: arms=blue, legs=orange, face=green
pub const KPT_COLOR_INDICES: [usize; 17] = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0];

/// Draw pose estimation results (skeleton and keypoints)
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `result` - The inference results containing keypoints
/// * `skeleton` - Optional custom skeleton structure (pairs of keypoint indices). If `None`, uses
///   the default human pose skeleton from `SKELETON`.
/// * `limb_colors` - Optional custom color indices for limbs. If `None`, uses the default from
///   `LIMB_COLOR_INDICES`.
/// * `kpt_colors` - Optional custom color indices for keypoints. If `None`, uses the default from
///   `KPT_COLOR_INDICES`.
///
/// # Examples
///
/// ```ignore
/// // Use default human pose configuration
/// draw_pose(&mut img, result, None, None, None);
///
/// // Use custom skeleton for animals
/// const ANIMAL_SKELETON: [[usize; 2]; 10] = [...];
/// const ANIMAL_LIMB_COLORS: [usize; 10] = [0, 0, 9, 9, ...];
/// const ANIMAL_KPT_COLORS: [usize; 15] = [16, 16, 0, 0, ...];
/// draw_pose(
///     &mut img,
///     result,
///     Some(&ANIMAL_SKELETON),
///     Some(&ANIMAL_LIMB_COLORS),
///     Some(&ANIMAL_KPT_COLORS),
/// );
/// ```
pub fn draw_pose(
    img: &mut RgbImage,
    result: &ul::Results,
    skeleton: Option<&[[usize; 2]]>,
    limb_colors: Option<&[usize]>,
    kpt_colors: Option<&[usize]>,
) {
    let keypoints = match &result.keypoints {
        Some(kpts) => kpts,
        None => return,
    };

    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and radius
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let radius = (3.0 * scale_factor).round() as i32;

    // Use provided parameters or defaults
    let skeleton = skeleton.unwrap_or(&SKELETON);
    let limb_colors = limb_colors.unwrap_or(&LIMB_COLOR_INDICES);
    let kpt_colors = kpt_colors.unwrap_or(&KPT_COLOR_INDICES);

    let kpt_data = &keypoints.data;
    let n_persons = kpt_data.shape()[0];
    let n_kpts = kpt_data.shape()[1];

    for person_idx in 0..n_persons {
        for (limb_idx, &[kpt_a, kpt_b]) in skeleton.iter().enumerate() {
            if kpt_a >= n_kpts || kpt_b >= n_kpts {
                continue;
            }

            let x1 = kpt_data[[person_idx, kpt_a, 0]];
            let y1 = kpt_data[[person_idx, kpt_a, 1]];
            let conf1 = kpt_data[[person_idx, kpt_a, 2]];
            let x2 = kpt_data[[person_idx, kpt_b, 0]];
            let y2 = kpt_data[[person_idx, kpt_b, 1]];
            let conf2 = kpt_data[[person_idx, kpt_b, 2]];

            if conf1 > 0.5 && conf2 > 0.5 {
                let color_idx = limb_colors[limb_idx % limb_colors.len()];
                let color = Rgb(POSE_COLORS[color_idx]);
                draw_line_segment(img, x1, y1, x2, y2, color, thickness);
            }
        }

        for kpt_idx in 0..n_kpts {
            let x = kpt_data[[person_idx, kpt_idx, 0]];
            let y = kpt_data[[person_idx, kpt_idx, 1]];
            let conf = kpt_data[[person_idx, kpt_idx, 2]];

            if conf > 0.5 && x >= 0.0 && y >= 0.0 && x < width as f32 && y < height as f32 {
                let color_idx = kpt_colors[kpt_idx % kpt_colors.len()];
                let color = Rgb(POSE_COLORS[color_idx]);
                draw_filled_circle(img, x as i32, y as i32, radius, color);
            }
        }
    }
}
