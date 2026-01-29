use indicatif::ProgressStyle;

/// Get a standardized progress bar style
pub fn progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}: {wide_bar:.cyan/blue} {pos}/{len} [{elapsed_precise}]")
        .unwrap()
}
