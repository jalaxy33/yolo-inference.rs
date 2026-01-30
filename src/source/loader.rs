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
}

#[derive(Debug)]
pub struct SourceLoader {
    current_idx: usize,
    frames: Vec<FrameData>,
    len: usize,
}

impl SourceLoader {
    pub fn new(source: &Source) -> Result<Self> {
        let frames = match source {
            Source::ImagePath(path) => {
                if is_image_file(path) {
                    vec![FrameData::Path(path.clone())]
                } else {
                    vec![]
                }
            }
            Source::Directory(dir_path) => collect_images_from_dir(dir_path)?
                .into_iter()
                .map(FrameData::Path)
                .collect(),
            Source::ImagePathVec(paths) => paths
                .iter()
                .filter(|p| is_image_file(p))
                .cloned()
                .map(FrameData::Path)
                .collect(),
            Source::Image(img) => vec![FrameData::Image(img.clone())],
            Source::ImageVec(imgs) => imgs.iter().cloned().map(FrameData::Image).collect(),
        };
        let len = frames.len();

        Ok(Self {
            current_idx: 0,
            frames,
            len,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }
}

impl Iterator for SourceLoader {
    type Item = (DynamicImage, SourceMeta);

    /// Get the next image and its metadata (in lazy loading manner)
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.len {
            return None;
        }

        let frame_data = &self.frames[self.current_idx];
        let (image, source_path) = match frame_data {
            FrameData::Path(p) => match image::open(p) {
                Ok(img) => (img, Some(p.clone())),
                Err(e) => {
                    tracing::error!("Failed to open image: {:?}. Error: {}", p, e);
                    self.current_idx += 1;
                    return self.next();
                }
            },
            FrameData::Image(img) => (img.clone(), None),
        };

        let meta = SourceMeta {
            frame_idx: self.current_idx,
            total_frames: self.len,
            source_path,
        };

        self.current_idx += 1;
        Some((image, meta))
    }
}

/// Implement ExactSizeIterator (to use indicatif's ProgressIterator)
impl ExactSizeIterator for SourceLoader {
    fn len(&self) -> usize {
        self.len
    }
}
