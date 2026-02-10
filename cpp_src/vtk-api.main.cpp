#include <filesystem>
#include <iostream>

#include "path_utils.h"
#include "yolo_common.h"

#ifdef ENABLE_VTK
#include "vtk_utils.h"
#include "yolo_vtk_utils.h"
#endif

using namespace std;
using namespace filesystem;

void test_empty_image() {
    cout << "\n=== Testing empty image handling ===" << endl;

#ifdef ENABLE_VTK
    // Create an empty RustImage (0x0 size)
    auto empty_image = yolo_inference::image_from_bytes(nullptr, 0, 0, 0);

    // Convert to VTK - should return nullptr
    auto vtk_image = rust2vtk(*empty_image);

    if (vtk_image) {
        cerr << "FAIL: Expected nullptr for empty image, but got valid vtkImageData" << endl;
    } else {
        cout << "PASS: Empty image correctly returns nullptr" << endl;
    }
#else
    cout << "VTK support is disabled, skipping empty image test" << endl;
#endif
}

void run_batch_prediction(const path& image_dir, const path& config_toml,
                          const path& save_dir = path()) {
    cout << "\n=== Running batch prediction ===" << endl;

    assert_path_exists(image_dir);
    assert_path_exists(config_toml);

    vector<path> image_paths = list_image_paths(image_dir);
    cout << "Found " << image_paths.size() << " image files." << endl;

    if (!save_dir.empty()) {
        clean_and_create_dir(save_dir);
    }

#ifdef ENABLE_VTK
    // Load images as vtkImageData
    vector<Ptr<vtkImageData>> vtk_images = gather_vtk_images(image_paths);
    cout << "\nLoaded " << vtk_images.size() << " vtkImageData(s) successfully." << endl;

    // Convert loaded vtkImageData to RustImage
    Vec<Box<RustImage>> rs_images = multi_vtk2rust(vtk_images);
    cout << "Converted " << rs_images.size() << " vtkImageData(s) to RustImage(s)." << endl;

    // Run inference on RustImages
    path project_root = PROJECT_ROOT;
    Vec<Box<InferResult>> results = yolo_inference::online_predict_from_toml(
        std::move(rs_images), config_toml.string(), project_root.string());

    // Get annotated images from results
    vector<Ptr<vtkImageData>> annotateds = get_batch_annotated(results);
    cout << "\nObtained " << annotateds.size() << " annotated vtkImageData(s)." << endl;

    // Optionally save annotated images
    if (!save_dir.empty()) {
        cout << "\nSaving annotated images to: " << save_dir << endl;
        for (size_t i = 0; i < annotateds.size() && i < image_paths.size(); ++i) {
            path save_path = save_dir / (image_paths[i].stem().string() + ".png");
            save_vtk_image(annotateds[i], save_path);
            cout << "  Saved: " << save_path.filename() << endl;
        }
    }

    cout << "\nBatch prediction completed." << endl;
#else   // !ENABLE_VTK
    cerr << "\nVTK support is disabled. Skipping VTK-related codes." << endl;
    return;
#endif  // ENABLE_VTK
}

int main() {
    path project_root = PROJECT_ROOT;
    path config_toml = project_root / "assets/configs/online-predict.toml";
    path image_dir = project_root / "assets/images/small-batch";
    path save_dir = project_root / "results/vtk-api";

    assert_path_exists(config_toml);
    assert_path_exists(image_dir);

    cout << "Using config: " << config_toml << endl;
    cout << "Using image directory: " << image_dir << endl;

    test_empty_image();

    run_batch_prediction(image_dir, config_toml, save_dir);

    return 0;
}
