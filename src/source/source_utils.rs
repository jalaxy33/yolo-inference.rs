use anyhow::Result;
use std::path::PathBuf;

pub fn is_image_file(path: &PathBuf) -> bool {
    path.extension().is_some_and(|ext| {
        let ext = ext.to_string_lossy().to_lowercase();
        matches!(
            ext.as_str(),
            "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif"
        )
    })
}

pub fn collect_images_from_dir(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut image_paths = vec![];
    for entry in std::fs::read_dir(dir)? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_file() && is_image_file(&path) {
                image_paths.push(path);
            }
        }
    }
    Ok(image_paths)
}
