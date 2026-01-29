/// Generates python stub files (.pyi) for the FFI module.

fn main() {
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let stub = yolo_inference::stub_info().expect("Failed to get stub info");
    stub.generate().expect("Failed to generate stub");
}
