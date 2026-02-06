// C++ example for calling Rust API
#include <filesystem>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "cpp_ffi.rs.h"
#include "stb_image.h"

using namespace std;
using namespace filesystem;

enum Experiment {
    OneImage,
    SmallBatch,
    LargeBatch,
    UnbatchableModel,
    OnlinePredict,
};

bool assert_path_exists(const path& p) {
    if (!exists(p)) {
        cerr << "Error: Path does not exist: " << p << endl;
        exit(1);
    }
    return true;
}

path get_config_path(const path& config_dir, Experiment experiment) {
    switch (experiment) {
        case OneImage:
            return config_dir / "one-image.toml";
        case SmallBatch:
            return config_dir / "small-batch.toml";
        case LargeBatch:
            return config_dir / "large-batch.toml";
        case UnbatchableModel:
            return config_dir / "unbatchable-model.toml";
        case OnlinePredict:
            return config_dir / "online-predict.toml";
        default:
            throw runtime_error("Unknown experiment");
    }
}

void run_online_predict(const path& config_toml) {
    cout << "Running online prediction..." << endl;

    // Load test images from small-batch directory
    path project_root = PROJECT_ROOT;
    vector<path> image_paths = {
        project_root / "assets/images/small-batch/boats.jpg",
        project_root / "assets/images/small-batch/bus.jpg",
        project_root / "assets/images/small-batch/zidane.jpg"};

    rust::Vec<rust::Box<RustImage>> images;

    for (const auto& img_path : image_paths) {
        if (!exists(img_path)) {
            cerr << "Warning: Image not found: " << img_path << endl;
            continue;
        }

        // Load image via stb_image
        int width, height, channels;
        unsigned char* pixels =
            stbi_load(img_path.c_str(), &width, &height, &channels, 0);

        if (!pixels) {
            cerr << "Failed to load image: " << img_path << endl;
            continue;
        }

        cout << "Loaded: " << img_path.filename() << " (" << width << "x"
             << height << ", " << channels << " channels)" << endl;

        // Pass raw pixels to Rust FFI (supports 1/3/4 channels)
        if (channels == 1 || channels == 3 || channels == 4) {
            rust::Box<RustImage> rust_img =
                image_from_pixels(pixels, width, height, channels);

            ImageInfo info = image_info(*rust_img);
            cout << "  -> RustImage: " << info.width << "x" << info.height
                 << ", channels=" << info.channels << endl;

            images.push_back(std::move(rust_img));
        } else {
            cerr << "  -> Skipped: unsupported channel count " << channels
                 << endl;
        }

        stbi_image_free(pixels);
    }

    cout << "\nTotal images loaded: " << images.size() << endl;

    if (images.empty()) {
        cerr << "No valid images to process!" << endl;
        return;
    }

    // Run online inference via Rust FFI
    cout << "\nRunning online prediction..." << endl;
    rust::Vec<rust::Box<InferResult>> results =
        online_predict_from_toml(std::move(images), config_toml.string());

    cout << "Prediction completed. Results count: " << results.size() << endl;

    // Test get_result_annotated (clone version)
    cout << "\nTesting get_result_annotated (clone version):" << endl;
    for (size_t i = 0; i < results.size(); i++) {
        // rust::Box automatically manages memory
        rust::Box<RustImage> annotated = get_result_annotated(*results[i]);

        // Test is_image_empty function
        bool empty = is_image_empty(*annotated);
        ImageInfo info = image_info(*annotated);
        cout << "  Result[" << i << "] annotated image: " << info.width
             << "x" << info.height << ", channels=" << info.channels
             << ", empty=" << (empty ? "true" : "false") << endl;
    }

    // Test take_result_annotated (ownership transfer version)
    cout << "\nTesting take_result_annotated (take version):" << endl;
    for (size_t i = 0; i < results.size(); i++) {
        rust::Box<RustImage> annotated = take_result_annotated(*results[i]);

        bool empty = is_image_empty(*annotated);
        ImageInfo info = image_info(*annotated);
        cout << "  Result[" << i << "] took annotated image: " << info.width
             << "x" << info.height << ", channels=" << info.channels
             << ", empty=" << (empty ? "true" : "false") << endl;
    }
}

int main() {
    path project_root = PROJECT_ROOT;
    path config_dir = project_root / "assets/configs";
    assert_path_exists(project_root);

    Experiment experiment = OnlinePredict;
    path config_toml = get_config_path(config_dir, experiment);
    assert_path_exists(config_toml);

    cout << "Using config: " << config_toml << endl;

    if (experiment == OnlinePredict) {
        run_online_predict(config_toml);
    } else {
        // Run prediction
        predict_from_toml(config_toml);
    }

    return 0;
}
