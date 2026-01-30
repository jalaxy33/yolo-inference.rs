use image::DynamicImage;
use indicatif::{ProgressBar, ProgressFinish};
use std::sync::mpsc;
use std::thread;
use ultralytics_inference as ul;

use crate::annotate::annotate_image;
use crate::error::Result;
use crate::predict::PredictArgs;
use crate::progress_bar::progress_bar_style;
use crate::source::{SourceLoader, SourceMeta};

use super::InferResult;

/// Channel-based concurrent pipeline inference
///
/// - To avoid huge memory consumption, results are returned via `return_results` argument if
///   provided.
pub fn channel_pipeline_infer(
    model: &mut ul::YOLOModel,
    args: &PredictArgs,
    return_results: &mut Option<Vec<InferResult>>,
) -> Result<()> {
    let source = &args.source;
    let annotate = args.annotate;
    let annotate_cfg = &args.annotate_cfg;
    let save_dir = &args.save_dir;
    let channel_capacity = args.channel_capacity.unwrap_or(8);
    let verbose = args.verbose;
    let save = annotate && save_dir.is_some();

    tracing::info!("Running channel-based pipeline inference...");
    tracing::info!("[Source]: {:?}", args.source);

    // Prepare save directory
    if let Some(dir) = save_dir {
        if dir.is_dir() {
            tracing::warn!("Clearing existing save directory: {:?}", dir);
            std::fs::remove_dir_all(dir).expect("Failed to clear existing save directory");
        }
        std::fs::create_dir_all(dir).expect("Failed to create save directory");
    }
    // Initialize source loader
    let loader = SourceLoader::new(source);
    let total_frames = loader.len();
    tracing::info!("Total frames to process: {}", total_frames);
    tracing::info!("-----------------------------------------");

    // preserve space in return_results if provided
    if let Some(vec) = return_results.as_mut() {
        vec.clear();
        vec.reserve(total_frames);
    }

    // Define data types for each pipeline stage
    type LoadStage = (DynamicImage, SourceMeta);
    type InferStage = (DynamicImage, ul::Results, SourceMeta);
    type AnnotateStage = (Option<DynamicImage>, ul::Results, SourceMeta);
    type SaveStage = (Option<DynamicImage>, ul::Results, SourceMeta);

    // Create channels for pipeline stages with bounded capacity
    let (load_tx, load_rx) = mpsc::sync_channel::<LoadStage>(channel_capacity);
    let (infer_tx, infer_rx) = mpsc::sync_channel::<InferStage>(channel_capacity);
    let (annotate_tx, annotate_rx) = mpsc::sync_channel::<AnnotateStage>(channel_capacity);
    let (save_tx, save_rx) = mpsc::sync_channel::<SaveStage>(channel_capacity);

    // initialize progress bar
    let pb = ProgressBar::new(total_frames as u64)
        .with_style(progress_bar_style())
        .with_message("Running inference")
        .with_finish(ProgressFinish::WithMessage("Finished".into()));

    // Use scoped threads to allow borrowing model
    thread::scope(|s| {
        // Stage 1: Image Loading thread
        let load_handler = s.spawn(move || {
            for (image, meta) in loader {
                if verbose {
                    tracing::debug!("[Loading]: {}", &meta.frame_name());
                }

                // Send loaded data to model inference stage
                if load_tx.send((image, meta)).is_err() {
                    break;
                }
            }
        });

        // Stage 2: Model inference thread
        let infer_handler = s.spawn(move || {
            while let Ok((image, meta)) = load_rx.recv() {
                if verbose {
                    tracing::debug!("[Inferring]: {}", &meta.frame_name());
                }
                let results_vec = match model.predict_image(&image, "".to_string()) {
                    Ok(res) => res,
                    Err(e) => {
                        tracing::error!(
                            "Prediction failed for image: {:?}, skipping. Error: {}",
                            &meta.source_path,
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
                            &meta.source_path
                        );
                        continue;
                    }
                };
                // Send inference results to annotation stage
                if infer_tx.send((image, results, meta)).is_err() {
                    break;
                }
            }
        });

        // Stage 3: Draw annotation thread
        let annotate_handler = s.spawn(move || {
            while let Ok((image, results, meta)) = infer_rx.recv() {
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
                // Send annotated image to saving stage
                if annotate_tx.send((annotated_img, results, meta)).is_err() {
                    break;
                }
            }
        });

        // Stage 4: Saving thread
        let save_handler = s.spawn(move || {
            while let Ok((annotated_img, results, meta)) = annotate_rx.recv() {
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

                if save_tx.send((annotated_img, results, meta)).is_err() {
                    break;
                }
            }
        });

        // Stage 5: Collect results thread
        let collect_handler = s.spawn(move || {
            while let Ok((annotated_img, results, meta)) = save_rx.recv() {
                // Update return results vector if provided
                if let Some(vec) = return_results {
                    if verbose {
                        tracing::debug!("[Collecting] result for: {}", &meta.frame_name());
                    }

                    vec.push(InferResult {
                        result: results,
                        annotated: annotated_img,
                        meta,
                    });
                }

                // Update progress bar
                pb.inc(1);
            }
        });

        // Wait for pipeline threads to finish
        load_handler.join().expect("Loading thread panicked");
        infer_handler.join().expect("Inference thread panicked");
        annotate_handler.join().expect("Annotation thread panicked");
        save_handler.join().expect("Saving thread panicked");
        collect_handler.join().expect("Collect thread panicked");
    });

    if save {
        tracing::info!(
            "Results saved to directory: {:?}",
            save_dir.as_ref().unwrap()
        );
    }
    Ok(())
}
