use indicatif::{ProgressFinish, ProgressIterator};
use ultralytics_inference as ul;

use crate::annotate::annotate_image;
use crate::error::Result;
use crate::predict::PredictArgs;
use crate::progress_bar::progress_bar_style;
use crate::source::SourceLoader;

use super::InferResult;

/// Naive sequential inference: process images one by one
///
/// - To avoid huge memory consumption, results are returned via `return_results` argument if
///   provided.
pub fn sequential_infer(
    model: &mut ul::YOLOModel,
    args: &PredictArgs,
    return_results: &mut Option<Vec<InferResult>>,
) -> Result<()> {
    let source = &args.source;
    let annotate = args.annotate;
    let annotate_cfg = &args.annotate_cfg;
    let save_dir = &args.save_dir;
    let verbose = args.verbose;
    let save = annotate && save_dir.is_some();

    tracing::info!("Running naive sequential inference...");
    tracing::info!("[Source]: {:?}", args.source);

    if let Some(dir) = save_dir {
        if dir.is_dir() {
            tracing::warn!("Clearing existing save directory: {:?}", dir);
            std::fs::remove_dir_all(dir).expect("Failed to clear existing save directory");
        }
        std::fs::create_dir_all(dir).expect("Failed to create save directory");
    }

    let loader = SourceLoader::new(source)?;
    let total_frames = loader.len();
    tracing::info!("Total frames to process: {}", total_frames);
    tracing::info!("-----------------------------------------");

    // preserve space in return_results if provided
    if let Some(vec) = return_results.as_mut() {
        vec.clear();
        vec.reserve(total_frames);
    }

    for (idx, (image, meta)) in loader
        .enumerate()
        .progress_with_style(progress_bar_style())
        .with_message("Running inference")
        .with_finish(ProgressFinish::WithMessage("Finished".into()))
    {
        if verbose {
            match &meta.source_path {
                Some(p) => {
                    tracing::debug!("Processing: {:?}", p.file_name().unwrap_or_default())
                }
                None => tracing::debug!("Processing index: {}", idx),
            }
        }

        let results_vec = match model.predict_image(&image, "".to_string()) {
            Ok(res) => res,
            Err(e) => {
                tracing::error!(
                    "Prediction failed for image: {:?}, skipping. Error: {}",
                    meta.source_path,
                    e
                );
                continue;
            }
        };

        // One image at a time
        let results = match results_vec.into_iter().next() {
            Some(r) => r,
            None => {
                tracing::error!(
                    "No results returned for image: {:?}, skipping.",
                    meta.source_path
                );
                continue;
            }
        };

        // draw annotations
        let annotated_img = if annotate {
            match annotate_image(&image, &results, annotate_cfg) {
                Ok(img) => Some(img),
                Err(e) => {
                    tracing::error!(
                        "Annotation failed for image: {:?}, skipping. Error: {}",
                        &meta.source_path,
                        e
                    );
                    continue;
                }
            }
        } else {
            None
        };

        // Save results if save_dir is specified
        if let Some(dir) = save_dir
            && let Some(annotated_img) = &annotated_img
        {
            let frame_stem = meta.frame_stem();
            let save_path = dir.join(format!("{}.png", frame_stem));
            if annotated_img.save(&save_path).is_err() {
                tracing::error!(
                    "Failed to save annotated image to {:?}. skipping.",
                    save_path
                );
                continue;
            }
        }

        // Update return results vector if provided
        if let Some(vec) = return_results {
            vec.push(InferResult {
                result: results,
                annotated: annotated_img,
                meta,
            });
        }
    }

    if save {
        tracing::info!(
            "Results saved to directory: {:?}",
            save_dir.as_ref().unwrap()
        );
    }
    Ok(())
}
