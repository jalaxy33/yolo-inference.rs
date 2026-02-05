use indicatif::{ProgressBar, ProgressFinish};
use ultralytics_inference as ul;

use crate::annotate::annotate_image;
use crate::error::Result;
use crate::predict::PredictArgs;
use crate::progress_bar::progress_bar_style;
use crate::source::{BatchSourceLoader, Source};

use super::InferResult;
use super::batch_utils::{batch_infer_fallback, get_batch_frame_names};

/// Sequential batch inference.
///
/// - To avoid huge memory consumption, results are returned via `return_results` argument if
///  provided.
pub fn batch_sequential_infer(
    model: &mut ul::YOLOModel,
    source: &Source,
    args: &PredictArgs,
    return_results: &mut Option<Vec<InferResult>>,
) -> Result<()> {
    let annotate = args.annotate;
    let annotate_cfg = &args.annotate_cfg;
    let save_dir = &args.save_dir;
    let batch_size = args.batch.unwrap_or(1);
    let verbose = args.verbose;
    let save = annotate && save_dir.is_some();

    tracing::info!("Running sequential batch inference...");
    tracing::info!("[Source]: {:?}", source);
    tracing::info!("Batch Size: {}", batch_size);

    if let Some(dir) = save_dir {
        if dir.is_dir() {
            tracing::warn!("Clearing existing save directory: {:?}", dir);
            std::fs::remove_dir_all(dir).expect("Failed to clear existing save directory");
        }
        std::fs::create_dir_all(dir).expect("Failed to create save directory");
    }

    let loader = BatchSourceLoader::new(source, Some(batch_size))?;
    let total_batches = loader.len();
    let total_frames = loader.total_frames();
    tracing::info!("Total batches to process: {}", total_batches);
    tracing::info!("Total frames to process: {}", total_frames);
    tracing::info!("-----------------------------------------");

    // preserve space in return_results if provided
    if let Some(vec) = return_results.as_mut() {
        vec.clear();
        vec.reserve(total_frames);
    }

    let pseudo_paths = vec!["".to_string(); batch_size];

    // initialize progress bar
    let pb = ProgressBar::new(total_frames as u64)
        .with_style(progress_bar_style())
        .with_message("Running inference")
        .with_finish(ProgressFinish::WithMessage("Finished".into()));

    // record if batch inference has failed before
    let mut infer_failed = false;

    for (batch_idx, (batch_images, batch_metas)) in loader.enumerate() {
        if verbose {
            let frame_names = get_batch_frame_names(&batch_metas);
            tracing::debug!("Processing batch {}: {:?}", batch_idx, frame_names);
        }

        // Try to predict batch,
        // if fails, try to predict images one by one
        let batch_results: Vec<Option<ul::Results>> = if !infer_failed {
            match model.predict_batch(&batch_images, &pseudo_paths) {
                Ok(vec) => {
                    // try to extract first element from each
                    vec.into_iter().map(|mut v| Some(v.remove(0))).collect()
                }
                Err(e) => {
                    // Fallback to one by one inference
                    tracing::warn!(
                        "Batch inference failed for batch {}, falling back to one-by-one inference.",
                        batch_idx,
                    );
                    tracing::error!("> Error details: {:?}", e);
                    infer_failed = true;
                    batch_infer_fallback(model, &batch_images, &batch_metas, verbose)
                }
            }
        } else {
            // Fallback to one by one inference
            batch_infer_fallback(model, &batch_images, &batch_metas, verbose)
        };

        for (i, results) in batch_results.into_iter().enumerate() {
            // skip invalid results
            let results = match results {
                Some(r) => r,
                None => {
                    continue;
                }
            };

            let image = &batch_images[i];
            let meta = &batch_metas[i];

            // draw annotations
            let annotated_img = if annotate {
                if verbose {
                    tracing::debug!("[Annotating]: {}", &meta.frame_name());
                }

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

            // Save annotated image if required
            if let Some(dir) = save_dir
                && let Some(annotated_img) = &annotated_img
            {
                if verbose {
                    tracing::debug!("[Saving]: {}", &meta.frame_name());
                }

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

            // Store results if required
            if let Some(vec) = return_results.as_mut() {
                if verbose {
                    tracing::debug!("[Collecting] results for: {}", &meta.frame_name());
                }

                vec.push(InferResult {
                    result: results,
                    annotated: annotated_img,
                    meta: meta.clone(),
                });
            }

            // update progress bar
            pb.inc(1);
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
