use std::fs::File;
use std::io::Read;
use ultralytics_inference as ul;

/// Helper to check if a string contains non-ASCII characters
pub const fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

pub fn load_font(font_name: &str) -> Option<Vec<u8>> {
    let font_path = ul::annotate::check_font(font_name);
    let font_data = font_path.and_then(|path| {
        let mut file = File::open(path).ok()?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).ok()?;
        Some(buffer)
    });
    font_data
}
