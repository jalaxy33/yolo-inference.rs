use image::DynamicImage;
use std::iter::ExactSizeIterator;
use std::path::PathBuf;

use crate::error::Result;

use super::source_utils::{collect_images_from_dir, is_image_file};
use super::{Source, SourceMeta};

#[derive(Debug, Clone)]
enum FrameData {
    Path(PathBuf),
    Image(DynamicImage),
    None, // Padding for incomplete batches
}

#[derive(Debug)]
pub struct BatchSourceLoader {
    current_idx: usize,
    batches: Vec<Vec<FrameData>>,
    len: usize,
    batch_size: usize,
    total_frames: usize,
}

impl BatchSourceLoader {
    pub fn new(source: &Source, batch_size: Option<usize>) -> Result<Self> {
        let batch_size = match batch_size {
            Some(size) if size > 0 => size,
            _ => 1,
        };

        let (batches, num_pads) = match source {
            Source::ImagePath(p) => {
                if is_image_file(p) {
                    (vec![vec![FrameData::Path(p.clone())]], 0)
                } else {
                    (vec![], 0)
                }
            }
            Source::Directory(dir_path) => {
                let frames_vec: Vec<FrameData> = collect_images_from_dir(dir_path)?
                    .into_iter()
                    .map(FrameData::Path)
                    .collect();
                Self::pad_and_chunk(frames_vec, batch_size)
            }
            Source::ImagePathVec(paths) => {
                let frames_vec: Vec<FrameData> = paths
                    .iter()
                    .filter(|p| is_image_file(p))
                    .cloned()
                    .map(FrameData::Path)
                    .collect();
                Self::pad_and_chunk(frames_vec, batch_size)
            }
            Source::Image(img) => (vec![vec![FrameData::Image(img.clone())]], 0),
            Source::ImageVec(imgs) => {
                let frames_vec: Vec<FrameData> =
                    imgs.iter().cloned().map(FrameData::Image).collect();
                Self::pad_and_chunk(frames_vec, batch_size)
            }
        };
        let len = batches.len();
        let total_frames = len * batch_size - num_pads;
        Ok(Self {
            current_idx: 0,
            batches,
            len,
            batch_size,
            total_frames,
        })
    }

    fn pad_and_chunk(
        frames_vec: Vec<FrameData>,
        batch_size: usize,
    ) -> (Vec<Vec<FrameData>>, usize) {
        let mut frames_vec = frames_vec;
        let mut num_pads = 0;
        while frames_vec.len() % batch_size != 0 {
            frames_vec.push(FrameData::None);
            num_pads += 1;
        }
        let chunks = frames_vec
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        (chunks, num_pads)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn total_frames(&self) -> usize {
        self.total_frames
    }
}

impl Iterator for BatchSourceLoader {
    type Item = (Vec<DynamicImage>, Vec<SourceMeta>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.len {
            return None;
        }

        let batch_frames = &self.batches[self.current_idx];
        let mut batch_images = Vec::with_capacity(self.batch_size);
        let mut batch_metas = Vec::with_capacity(self.batch_size);

        for (i, frame_data) in batch_frames.iter().enumerate() {
            match frame_data {
                FrameData::Path(p) => {
                    let img = match image::open(p) {
                        Ok(img) => img,
                        Err(e) => {
                            eprintln!("Failed to open image {:?}: {}", p, e);
                            continue;
                        }
                    };
                    let meta = SourceMeta {
                        frame_idx: self.current_idx * self.batch_size + i,
                        total_frames: self.len * self.batch_size,
                        source_path: Some(p.clone()),
                    };
                    batch_images.push(img);
                    batch_metas.push(meta);
                }
                FrameData::Image(img) => {
                    let meta = SourceMeta {
                        frame_idx: self.current_idx * self.batch_size + i,
                        total_frames: self.len * self.batch_size,
                        source_path: None,
                    };
                    // batch_data.push((img.clone(), meta));
                    batch_images.push(img.clone());
                    batch_metas.push(meta);
                }
                FrameData::None => {
                    // Skip padding frames
                }
            }
        }

        self.current_idx += 1;
        Some((batch_images, batch_metas))
    }
}

/// Implement ExactSizeIterator (to use indicatif's ProgressIterator)
impl ExactSizeIterator for BatchSourceLoader {
    fn len(&self) -> usize {
        self.len
    }
}
