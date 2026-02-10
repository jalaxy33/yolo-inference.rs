#include <filesystem>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "path_utils.h"
#include "stb_image.h"
#include "yolo-inference/src/ffi/cpp_ffi.rs.h"

using namespace std;
using namespace filesystem;

using yolo_inference::ImageInfo;
using yolo_inference::InferResult;
using yolo_inference::RustImage;

using rust::Box;
using rust::Vec;

Vec<Box<RustImage>> gather_rust_images(const vector<path>& image_paths) {
    Vec<Box<RustImage>> images;

    for (const auto& img_path : image_paths) {
        cout << "--------------------------------\n"
             << "Processing image: " << img_path.filename() << endl;

        if (!exists(img_path)) {
            cerr << "Warning: Image not found: " << img_path << endl;
            continue;
        }

        // Load image using stb_image
        int width, height, channels;
        unsigned char* bytes = stbi_load(img_path.c_str(), &width, &height, &channels, 0);

        if (!bytes) {
            cerr << "  Failed to load image: " << img_path << endl;
            continue;
        }

        cout << "  Loaded image: (" << width << "x" << height << ", " << channels << " channels)"
             << endl;

        // Gather RustImages via FFI
        if (channels == 1 || channels == 3 || channels == 4) {
            Box<RustImage> rust_img =
                yolo_inference::image_from_bytes(bytes, width, height, channels);
            ImageInfo info = yolo_inference::get_image_info(*rust_img);

            cout << "  -> RustImage: " << info.width << "x" << info.height
                 << ", channels=" << info.channels << endl;

            images.push_back(std::move(rust_img));
        } else {
            cerr << "  -> Skipped: unsupported channel count " << channels << endl;
        }

        stbi_image_free(bytes);
    }

    cout << "\nTotal valid images gathered: " << images.size() << endl;
    return images;
}

void test_get_annotated(const Vec<Box<InferResult>>& results) {
    cout << "\nTesting get_result_annotated (clone version):" << endl;
    for (size_t i = 0; i < results.size(); i++) {
        // rust::Box automatically manages memory
        Box<RustImage> annotated = yolo_inference::get_result_annotated(*results[i]);

        // Test is_image_empty function
        bool empty = is_image_empty(*annotated);
        ImageInfo info = get_image_info(*annotated);
        cout << "  Result[" << i << "] annotated image: " << info.width << "x" << info.height
             << ", channels=" << info.channels << ", empty=" << (empty ? "true" : "false") << endl;
    }
}

void test_take_annotated(Vec<Box<InferResult>> results) {
    cout << "\nTesting get_result_annotated (take version):" << endl;
    for (size_t i = 0; i < results.size(); i++) {
        // Take ownership of the annotated image
        Box<RustImage> annotated = yolo_inference::take_result_annotated(*results[i]);

        // Test is_image_empty function
        bool empty = is_image_empty(*annotated);
        ImageInfo info = get_image_info(*annotated);
        cout << "  Result[" << i << "] annotated image: " << info.width << "x" << info.height
             << ", channels=" << info.channels << ", empty=" << (empty ? "true" : "false") << endl;
    }
}

int main() {
    path project_root = PROJECT_ROOT;
    path config_toml = project_root / "assets/configs/online-predict.toml";
    path image_dir = project_root / "assets/images/small-batch";

    assert_path_exists(config_toml);
    assert_path_exists(image_dir);

    cout << "Using config: " << config_toml << endl;
    cout << "Using image directory: " << image_dir << endl;

    auto image_paths = list_image_paths(image_dir);
    cout << "Found " << image_paths.size() << " images." << endl;

    // Gather RustImages
    auto images = gather_rust_images(image_paths);

    // Run online inference via Rust FFI
    Vec<Box<InferResult>> results =
        yolo_inference::online_predict_from_toml(std::move(images), config_toml.string());

    cout << "\n--------------------------------\n"
         << "Prediction completed. Results count: " << results.size() << endl;

    // Get annotated results
    test_get_annotated(results);
    test_take_annotated(std::move(results));

    return 0;
}
