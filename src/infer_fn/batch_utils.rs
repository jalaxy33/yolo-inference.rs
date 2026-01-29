use image::DynamicImage;
use ultralytics_inference as ul;

use crate::source::SourceMeta;

/// Get frame names for a batch of source metas
pub fn get_batch_frame_names(batch_metas: &Vec<SourceMeta>) -> Vec<String> {
    let mut frame_names: Vec<String> = Vec::with_capacity(batch_metas.len());
    for (i, meta) in batch_metas.iter().enumerate() {
        match &meta.source_path {
            Some(p) => frame_names.push(
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            ),
            None => frame_names.push(format!("frame_{}", meta.frame_idx * batch_metas.len() + i)),
        }
    }
    frame_names
}

/// Fallback model inference stage via sequential per-image inference
pub fn batch_infer_fallback(
    model: &mut ul::YOLOModel,
    images: &[DynamicImage],
    metas: &[SourceMeta],
    verbose: bool,
) -> Vec<Option<ul::Results>> {
    let mut batch_results = Vec::with_capacity(images.len());
    for (image, meta) in images.iter().zip(metas.iter()) {
        if verbose {
            tracing::debug!("[Inferring] image: {}", &meta.frame_name());
        }

        let frame_results_vec = match model.predict_image(&image, "".to_string()) {
            Ok(res) => res,
            Err(e) => {
                tracing::error!(
                    "Prediction failed for image: {:?}, skipping. Error: {}",
                    &meta.source_path,
                    e
                );
                batch_results.push(None);
                continue;
            }
        };
        // One image at a time
        let frame_results = match frame_results_vec.into_iter().next() {
            Some(r) => r,
            None => {
                tracing::error!(
                    "No results returned for image: {:?}, skipping.",
                    &meta.source_path
                );
                batch_results.push(None);
                continue;
            }
        };
        batch_results.push(Some(frame_results));
    }

    batch_results
}
