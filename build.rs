fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Generate C++ bindings when cpp feature is enabled.
    #[cfg(feature = "cpp")]
    {
        let _ = cxx_build::bridge("src/ffi/cpp_ffi.rs");
        println!("cargo:rerun-if-changed=src/ffi/cpp_ffi.rs");
    }

    #[cfg(feature = "python")]
    {
        setup_python();
    }
}

/// Set up python virtual environment and link library
#[cfg(feature = "python")]
fn setup_python() {
    use std::path::Path;

    let venv_dir_str = std::env::var("VIRTUAL_ENV").unwrap_or(".venv".to_string());
    let venv_dir = Path::new(&venv_dir_str);
    let python_executable = if cfg!(windows) {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python")
    };

    // Initialize virtual environment if needed
    let should_init_venv = !venv_dir.exists() || !python_executable.exists();
    if should_init_venv {
        println!(
            "cargo:warning=Virtual environment not found at {:?}. Running `uv sync`...",
            venv_dir
        );
        std::process::Command::new("uv")
            .arg("sync")
            .status()
            .expect("Failed to execute `uv sync`");
    }

    // Set cargo rerun triggers
    println!("cargo:rerun-if-changed={}", venv_dir.to_str().unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        python_executable.to_str().unwrap()
    );

    assert!(
        python_executable.exists(),
        "Python executable not found at {:?}",
        python_executable
    );

    // Get Python base_exec_prefix (the uv-installed Python location)
    let output = std::process::Command::new(&python_executable)
        .args(&["-c", "import sys; print(sys.base_exec_prefix)"])
        .output()
        .expect("Failed to execute python command to get executable path");

    if !output.status.success() {
        println!("cargo:warning=Failed to get Python base_exec_prefix");
        return;
    }

    let base_prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();

    #[cfg(target_os = "windows")]
    {
        let path_env = std::env::var("PATH").unwrap_or_default();
        println!("cargo:rustc-env=PATH={};{}", path_env, base_prefix);
    }

    #[cfg(target_os = "linux")]
    {
        let lib_dir = Path::new(&base_prefix).join("lib");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search={}", lib_dir.to_str().unwrap());
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_dir.to_str().unwrap()
            );
        } else {
            println!(
                "cargo:warning=Python lib directory not found at {:?}",
                lib_dir
            );
        }
    }
}
