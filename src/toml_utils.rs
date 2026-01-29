// -- imports
use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;

use crate::annotate::AnnotateConfigs;
use crate::predict::PredictArgs;

// -- config

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct TomlConfig {
    predict: PredictArgs,
    annotate: AnnotateConfigs,
}

impl TomlConfig {
    /// Resolve relative paths against project root
    fn resolve_paths(mut self) -> Self {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Resolve model path
        if !self.predict.model.is_absolute() {
            self.predict.model = project_root.join(&self.predict.model);
        }

        // Resolve source path
        self.predict.source = match &self.predict.source {
            crate::source::Source::ImagePath(p) if !p.is_absolute() => {
                crate::source::Source::ImagePath(project_root.join(p))
            }
            crate::source::Source::Directory(p) if !p.is_absolute() => {
                crate::source::Source::Directory(project_root.join(p))
            }
            _ => self.predict.source,
        };

        // Resolve save_dir
        if let Some(ref mut save_dir) = self.predict.save_dir {
            if !save_dir.is_absolute() {
                *save_dir = project_root.join(save_dir.as_path());
            }
        }

        self
    }
}

// -- public API

pub fn parse_toml(toml_path: &PathBuf) -> Result<PredictArgs> {
    if !toml_path.is_file() || !toml_path.extension().map_or(false, |ext| ext == "toml") {
        anyhow::bail!(
            "TOML config path is not a valid .toml file: {:?}",
            toml_path
        );
    }

    let content = std::fs::read_to_string(toml_path)?;
    let config: TomlConfig = toml::from_str(&content)?;

    let config = config.resolve_paths();

    Ok(config.predict)
}
