use thiserror::Error;

/// Main error type for the library
#[derive(Error, Debug)]
pub enum AppError {
    #[error("TOML config file error: {0}")]
    TomlConfig(#[from] toml::de::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid device specification: {0}")]
    InvalidDevice(String),

    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Image loading failed: {0}")]
    ImageLoad(String),

    #[error("Image collection failed: {0}")]
    ImageCollection(String),

    #[error("Font loading failed: {0}")]
    FontLoad(String),

    #[error("YOLO inference error: {0}")]
    Inference(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Result type with default AppError
pub type Result<T, E = AppError> = std::result::Result<T, E>;
