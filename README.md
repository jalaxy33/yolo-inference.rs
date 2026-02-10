# yolo-inference.rs

A Rust YOLO inference library built on [ultralytics-rust](https://github.com/ultralytics/inference), offering Rust, C++, and Python bindings for efficient model inference.

## Quick Start

### Rust

```bash
# Build
cargo build -r

# Run
cargo run -r
```

### C++

```bash
# Build library with C++ bindings
cargo build -r --features cpp

# Build C++ project
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release

# Run (Linux/MacOS)
./build/Release/offline-predict
./build/Release/online-predict
./build/Release/vtk-api
```

### Python

```bash
# Build and install as python module
# in some platforms, you may need to use `go-task develop` instead
task develop

# Run demo
uv run python/main.py
```

## Example Configuration

```toml
[predict]
model = "checkpoints/yolo11n.onnx"
source = "images/"
conf = 0.25
iou = 0.45
batch = 8
device = "cuda:0"
infer_fn = "BatchChannelPipeline"
annotate = true
save_dir = "results/"

[annotate]
show_box = true
show_label = true
show_conf = true
```

## Inference Modes

| Mode                 | Description                          |
| -------------------- | ------------------------------------ |
| Sequential           | Single image processing              |
| BatchSequential      | Batch processing without parallelism |
| ChannelPipeline      | Multi-threaded pipeline              |
| BatchChannelPipeline | Batch + pipeline (default)           |

## Requirements

### Rust

- [rustup](https://rustup.rs/) - Rust toolchain manager
- nightly toolchain: `rustup toolchain install nightly`
- [cranelift](https://github.com/rust-lang/rustc_codegen_cranelift#download-using-rustup) - Code generator

### C++

- C++ compiler (g++ / clang++ / MSVC)
- [cmake](https://cmake.org/download/) - Build system

### Python

- [uv](https://docs.astral.sh/uv/) - Python package manager
- [task](https://taskfile.dev/) - Task runner
