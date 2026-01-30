use image::DynamicImage;
use indicatif::{ProgressBar, ProgressFinish};
use std::sync::mpsc;
use std::thread;
use ultralytics_inference as ul;

use crate::annotate::annotate_image;
use crate::error::Result;
use crate::predict::PredictArgs;
use crate::progress_bar::progress_bar_style;
use crate::source::{BatchSourceLoader, SourceMeta};

use super::InferResult;
use super::batch_utils::{batch_infer_fallback, get_batch_frame_names};

/// Channel-based concurrent pipeline for batch inference
///
/// - To avoid huge memory consumption, results are returned via `return_results` argument if
///   provided.
pub fn batch_channel_pipeline_infer(
    model: &mut ul::YOLOModel,
    args: &PredictArgs,
    return_results: &mut Option<Vec<InferResult>>,
) -> Result<()> {
    let source = &args.source;
    let annotate = args.annotate;
    let annotate_cfg = &args.annotate_cfg;
    let save_dir = &args.save_dir;
    let channel_capacity = args.channel_capacity.unwrap_or(8);
    let batch_size = args.batch.unwrap_or(1);
    let verbose = args.verbose;
    let save = annotate && save_dir.is_some();

    tracing::info!("Running channel-based batch pipeline inference...");
    tracing::info!("[Source]: {:?}", args.source);
    tracing::info!("Batch Size: {}", batch_size);

    if let Some(dir) = save_dir {
        if dir.is_dir() {
            tracing::warn!("Clearing existing save directory: {:?}", dir);
            std::fs::remove_dir_all(dir).expect("Failed to clear existing save directory");
        }
        std::fs::create_dir_all(dir).expect("Failed to create save directory");
    }

    let loader = BatchSourceLoader::new(source, Some(batch_size));
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

    // Define data types for each pipeline stage
    type LoadStage = (usize, Vec<DynamicImage>, Vec<SourceMeta>);
    type InferStage = (usize, DynamicImage, ul::Results, SourceMeta);
    type AnnotateStage = (usize, Option<DynamicImage>, ul::Results, SourceMeta);
    type SaveStage = (usize, Option<DynamicImage>, ul::Results, SourceMeta);

    // Create channels for each stage with bounded capacity
    let (load_tx, load_rx) = mpsc::sync_channel::<LoadStage>(channel_capacity);
    let (infer_tx, infer_rx) = mpsc::sync_channel::<InferStage>(channel_capacity);
    let (annotate_tx, annotate_rx) = mpsc::sync_channel::<AnnotateStage>(channel_capacity);
    let (save_tx, save_rx) = mpsc::sync_channel::<SaveStage>(channel_capacity);

    // initialize progress bar
    let pb = ProgressBar::new(total_frames as u64)
        .with_style(progress_bar_style())
        .with_message("Running inference")
        .with_finish(ProgressFinish::WithMessage("Finished".into()));

    // record if batch inference has failed before
    let mut infer_failed = false;

    // Use scoped threads to allow borrowing model
    thread::scope(|s| {
        // Stage 1: Image Loading thread
        let load_handle = s.spawn(move || {
            for (batch_idx, (batch_images, batch_metas)) in loader.enumerate() {
                if verbose {
                    let batch_frame_names = get_batch_frame_names(&batch_metas);
                    tracing::debug!("[Loading] batch {}: {:?}", batch_idx, batch_frame_names);
                }

                // Send loaded batch to next stage
                if load_tx
                    .send((batch_idx, batch_images, batch_metas))
                    .is_err()
                {
                    break;
                }
            }
        });

        // Stage 2: Model Inference thread
        let infer_handler = s.spawn(move || {
            while let Ok((batch_idx, batch_images, batch_metas)) = load_rx.recv() {
                if verbose {
                    let batch_frame_names = get_batch_frame_names(&batch_metas);
                    tracing::debug!("[Inferring] batch {}: {:?}", batch_idx, batch_frame_names);
                }

                let batch_results: Vec<Option<ul::Results>> = if !infer_failed {
                    match model
                        .predict_batch(&batch_images, &pseudo_paths)
                    {
                        Ok(vec) => {
                            // try to extract first element from each
                            vec.into_iter().map(|mut v| Some(v.remove(0))).collect()
                        }
                        Err(e) => {
                            // Failed first time, falling back to sequential per-image inference stage
                            tracing::warn!(
                                "Batch inference failed for batch {}, falling back to sequential per-image inference stage.",
                                batch_idx,
                            );
                            tracing::error!("> Error details: {:?}", e);
                            infer_failed = true;
                            batch_infer_fallback(model, &batch_images, &batch_metas, verbose)
                        }
                    }
                } else {
                    // Fallback to sequential per-image inference stage
                    batch_infer_fallback(model, &batch_images, &batch_metas, verbose)
                };

                // Send each valid inference result to next stage
                for ((image, result), meta) in batch_images
                    .into_iter()
                    .zip(batch_results.into_iter())
                    .zip(batch_metas.into_iter())
                {
                    if let Some(r) = result {
                        // Send inference results to next stage
                        if infer_tx.send((batch_idx, image, r, meta)).is_err() {
                            break;
                        }
                    }
                }

            }
        });

        // Stage 3: Annotation thread
        let annotate_handler = s.spawn(move || {
            while let Ok((batch_idx, image, results, meta)) = infer_rx.recv() {
                // draw annotations
                let annotated_img = if annotate {
                    if verbose {
                        tracing::debug!("[Annotating] batch {}: {}", batch_idx, &meta.frame_name());
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
                if annotate_tx
                    .send((batch_idx, annotated_img, results, meta))
                    .is_err()
                {
                    break;
                }
            }
        });

        // Stage 4: Saving thread
        let save_handler = s.spawn(move || {
            while let Ok((batch_idx, annotated_img, results, meta)) = annotate_rx.recv() {
                if let Some(dir) = save_dir
                    && let Some(annotated_img) = &annotated_img
                {
                    if verbose {
                        tracing::debug!("[Saving] batch {}: {}", batch_idx, &meta.frame_name());
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
                // Send to collection stage
                if save_tx
                    .send((batch_idx, annotated_img, results, meta))
                    .is_err()
                {
                    break;
                }
            }
        });

        // Stage 5: Collect results thread
        let collect_handler = s.spawn(move || {
            while let Ok((batch_idx, annotated_img, results, meta)) = save_rx.recv() {
                // Update return results vector if provided
                if let Some(vec) = return_results {
                    if verbose {
                        tracing::debug!("[Collecting] batch {}: {}", batch_idx, &meta.frame_name());
                    }

                    vec.push(InferResult {
                        result: results,
                        annotated: annotated_img,
                        meta: meta.clone(),
                    });
                }

                // Update progress bar
                pb.inc(1);
            }
        });

        // Wait for pipeline threads to finish
        load_handle.join().expect("Loading thread panicked");
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
