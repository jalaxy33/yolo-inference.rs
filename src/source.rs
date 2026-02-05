// -- submodules
mod batch_loader;
mod loader;
mod source_utils;

pub use batch_loader::BatchSourceLoader;
pub use loader::SourceLoader;
pub use source_utils::{collect_images_from_dir, is_image_file};

// -- external imports
use image::DynamicImage;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SourceMeta {
    /// Current frame index (0-based).
    pub frame_idx: usize,
    /// Total frames (1 for single images).
    pub total_frames: usize,
    /// Source path if available.
    pub source_path: Option<PathBuf>,
}

impl SourceMeta {
    pub fn frame_stem(&self) -> String {
        let frame_idx = self.frame_idx;
        let frame_stem = match &self.source_path {
            Some(p) => p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            None => format!("frame_{}", frame_idx),
        };
        frame_stem
    }

    pub fn frame_name(&self) -> String {
        let frame_idx = self.frame_idx;
        let frame_name = match &self.source_path {
            Some(p) => p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            None => format!("frame_{}", frame_idx),
        };
        frame_name
    }
}

#[derive(Clone)]
pub enum Source {
    /// No source provided (for online prediction)
    None,

    /// Path to a single image file
    ImagePath(PathBuf),

    /// Path to directory containing multiple images
    Directory(PathBuf),

    /// List of image paths
    ImagePathVec(Vec<PathBuf>),

    /// Image data in memory
    Image(DynamicImage),

    /// List of images in memory
    ImageVec(Vec<DynamicImage>),
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::None => write!(f, "None"),
            Source::ImagePath(p) => write!(f, "ImagePath({:?})", p),
            Source::Directory(p) => write!(f, "Directory({:?})", p),
            Source::ImagePathVec(v) => write!(f, "ImagePathVec({} items)", v.len()),
            Source::Image(img) => write!(f, "Image({}x{})", img.width(), img.height()),
            Source::ImageVec(v) => write!(f, "ImageVec({} items)", v.len()),
        }
    }
}

impl Source {
    /// Returns true if source is None
    pub fn is_none(&self) -> bool {
        matches!(self, Source::None)
    }

    pub fn is_batch(&self) -> bool {
        matches!(
            self,
            Source::Directory(_) | Source::ImagePathVec(_) | Source::ImageVec(_)
        )
    }

    pub fn is_image(&self) -> bool {
        matches!(self, Source::ImagePath(_) | Source::Image(_))
    }
}

impl From<PathBuf> for Source {
    fn from(path: PathBuf) -> Self {
        if path.is_dir() {
            Source::Directory(path)
        } else {
            Source::ImagePath(path)
        }
    }
}

impl From<&str> for Source {
    fn from(path: &str) -> Self {
        Source::from(PathBuf::from(path))
    }
}

impl From<String> for Source {
    fn from(path: String) -> Self {
        Source::from(PathBuf::from(path))
    }
}

impl From<Vec<PathBuf>> for Source {
    fn from(paths: Vec<PathBuf>) -> Self {
        Source::ImagePathVec(paths)
    }
}

impl From<DynamicImage> for Source {
    fn from(image: DynamicImage) -> Self {
        Source::Image(image)
    }
}

impl From<Vec<DynamicImage>> for Source {
    fn from(images: Vec<DynamicImage>) -> Self {
        Source::ImageVec(images)
    }
}

impl Default for Source {
    fn default() -> Self {
        Source::None
    }
}

/// Custom deserializer for Source from toml
/// Only supports PathBuf-based variants (ImagePath, Directory, ImagePathVec)
/// Empty string results in Source::None
pub fn deserialize_source<'de, D>(deserializer: D) -> Result<Source, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Try to deserialize as PathBuf first
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum PathOrVec {
        Path(PathBuf),
        Vec(Vec<PathBuf>),
    }

    match PathOrVec::deserialize(deserializer)? {
        PathOrVec::Path(path) => {
            // Empty path becomes None
            if path.as_os_str().is_empty() {
                Ok(Source::None)
            } else {
                Ok(path.into())
            }
        }
        PathOrVec::Vec(paths) => {
            // Empty vec becomes None
            if paths.is_empty() {
                Ok(Source::None)
            } else {
                Ok(paths.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_from_pathbuf() {
        // Test file path becomes ImagePath
        let path = PathBuf::from("test.jpg");
        let source: Source = path.clone().into();
        match source {
            Source::ImagePath(p) => assert_eq!(p, path),
            _ => panic!("Expected ImagePath"),
        }

        // Test directory path becomes Directory
        let path = PathBuf::from("/tmp");
        let source: Source = path.clone().into();
        match source {
            Source::Directory(p) => assert_eq!(p, path),
            _ => panic!("Expected Directory"),
        }
    }
}
