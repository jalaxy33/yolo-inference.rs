#[cfg(feature = "cpp")]
pub mod cpp_ffi;

#[cfg(feature = "python")]
pub mod python_ffi;

// -- export public API

#[cfg(feature = "python")]
pub use python_ffi::*;
