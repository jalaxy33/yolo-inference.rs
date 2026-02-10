// -- imports
use serde::Deserialize;
use std::path::Path;

use crate::annotate::AnnotateConfigs;
use crate::error::{AppError, Result};
use crate::predict::PredictArgs;
use crate::source::Source;

// -- config

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct TomlConfig {
    predict: PredictArgs,
    annotate: AnnotateConfigs,
}

impl TomlConfig {
    /// Parse TOML config file and return a TomlConfig instance.
    ///
    /// # Errors
    ///
    /// Returns `AppError` if:
    /// - The path is not a valid toml file
    /// - File read fails
    /// - TOML parsing fails
    /// Parse TOML config file with explicit project root for path resolution.
    ///
    /// # Arguments
    ///
    /// * `toml_path` - Path to the TOML config file
    /// * `project_root` - Base directory for resolving relative paths
    ///
    /// # Errors
    ///
    /// Returns `AppError` if:
    /// - The path is not a valid toml file
    /// - File read fails
    /// - TOML parsing fails
    pub fn from_toml(toml_path: &Path, project_root: &Path) -> Result<Self> {
        if !toml_path.is_file() || toml_path.extension().map_or(false, |ext| ext != "toml") {
            return Err(AppError::Config(format!(
                "TOML config path is not a valid .toml file: {:?}",
                toml_path
            )));
        }

        let content = std::fs::read_to_string(toml_path)?;
        let mut config: Self = toml::from_str(&content)?;
        config.resolve_paths(project_root);

        // Transfer annotate config to predict args
        config.predict.annotate_cfg = config.annotate.clone();

        Ok(config)
    }

    /// Resolve relative paths against project root
    fn resolve_paths(&mut self, project_root: &Path) {
        // Resolve model path
        if !self.predict.model.is_absolute() {
            self.predict.model = project_root.join(&self.predict.model);
        }

        // Resolve source path (skip if None)
        self.predict.source = match &self.predict.source {
            Source::None => Source::None,
            Source::ImagePath(p) if !p.is_absolute() => Source::ImagePath(project_root.join(p)),
            Source::Directory(p) if !p.is_absolute() => Source::Directory(project_root.join(p)),
            _ => self.predict.source.clone(),
        };

        // Resolve save_dir
        if let Some(ref mut save_dir) = self.predict.save_dir {
            if !save_dir.is_absolute() {
                *save_dir = project_root.join(save_dir.as_path());
            }
        }
    }
}

impl From<TomlConfig> for PredictArgs {
    fn from(config: TomlConfig) -> Self {
        config.predict
    }
}

// -- public API

/// Parse TOML config file and return PredictArgs.
///
/// # Arguments
///
/// * `toml_path` - Path to the TOML config file
/// * `project_root` - Base directory for resolving relative paths
///
/// # Errors
///
/// Returns `AppError` if TOML parsing or path resolution fails.
pub fn parse_toml(toml_path: &Path, project_root: &Path) -> Result<PredictArgs> {
    TomlConfig::from_toml(toml_path, project_root).map(Into::into)
}

// -- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_from_toml_with_custom_values() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("config.toml");
        let toml_content = r#"
[predict]
model = "test.onnx"
conf = 0.7
iou = 0.5
max_det = 100
half = true
annotate = true
verbose = true

[annotate]
on_blank = true
show_box = false
show_label = true
show_conf = false
top_k = 3
"#;
        fs::write(&toml_path, toml_content).unwrap();

        let config = TomlConfig::from_toml(&toml_path, temp_dir.path()).unwrap();

        assert_eq!(config.predict.conf, 0.7);
        assert_eq!(config.predict.iou, 0.5);
        assert_eq!(config.predict.max_det, 100);
        assert!(config.predict.half);
        assert!(config.predict.annotate);
        assert!(config.predict.verbose);

        assert!(config.annotate.on_blank);
        assert!(!config.annotate.show_box);
        assert!(config.annotate.show_label);
        assert!(!config.annotate.show_conf);
        assert_eq!(config.annotate.top_k, Some(3));
    }

    #[test]
    fn test_parse_toml_returns_predict_args() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("config.toml");
        let toml_content = r#"
[predict]
model = "test.onnx"
conf = 0.5

[annotate]
show_box = true
"#;
        fs::write(&toml_path, toml_content).unwrap();

        let args = parse_toml(&toml_path, temp_dir.path()).unwrap();

        assert_eq!(args.conf, 0.5);
        assert!(args.annotate_cfg.show_box);
    }

    #[test]
    fn test_from_toml_invalid_path() {
        let invalid_path = PathBuf::from("/nonexistent/config.toml");
        let project_root = PathBuf::from("/tmp");
        assert!(TomlConfig::from_toml(&invalid_path, &project_root).is_err());
    }

    #[test]
    fn test_from_toml_invalid_extension() {
        let temp_dir = TempDir::new().unwrap();
        let invalid_path = temp_dir.path().join("config.txt");
        fs::write(&invalid_path, "predict = { model = \"test.onnx\" }").unwrap();
        assert!(TomlConfig::from_toml(&invalid_path, temp_dir.path()).is_err());
    }

    #[test]
    fn test_parse_toml_invalid_toml() {
        let temp_dir = TempDir::new().unwrap();
        let invalid_toml_path = temp_dir.path().join("invalid.toml");
        fs::write(&invalid_toml_path, "invalid toml [[[").unwrap();
        assert!(parse_toml(&invalid_toml_path, temp_dir.path()).is_err());
    }

    #[test]
    fn test_from_toml_without_source_field() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("config.toml");
        let toml_content = r#"
[predict]
model = "test.onnx"
conf = 0.5

[annotate]
show_box = true
"#;
        fs::write(&toml_path, toml_content).unwrap();

        let config = TomlConfig::from_toml(&toml_path, temp_dir.path()).unwrap();

        // Source should be default (None)
        assert!(config.predict.source.is_none());
        assert_eq!(config.predict.conf, 0.5);
    }

    #[test]
    fn test_from_toml_with_empty_source() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("config.toml");
        let toml_content = r#"
[predict]
model = "test.onnx"
source = ""
conf = 0.5

[annotate]
show_box = true
"#;
        fs::write(&toml_path, toml_content).unwrap();

        let config = TomlConfig::from_toml(&toml_path, temp_dir.path()).unwrap();

        // Empty source string should become None
        assert!(config.predict.source.is_none());
        assert_eq!(config.predict.conf, 0.5);
    }
}
