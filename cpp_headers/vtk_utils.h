#pragma once

#include <filesystem>
#include <vector>

#ifdef ENABLE_VTK
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkImageReader2Factory.h>
#include <vtkSmartPointer.h>
#include <vtkPNGWriter.h>

using namespace std;
using namespace filesystem;

// Alias for VTK smart pointer
template <typename T>
using Ptr = vtkSmartPointer<T>;

/// Load `vtkImageData` from file path
inline Ptr<vtkImageData> load_vtk_image(const path& image_path) {
    if (!exists(image_path) || !is_regular_file(image_path)) {
        return nullptr;
    }

    auto reader = vtkSmartPointer<vtkImageReader2>::Take(
        vtkImageReader2Factory::CreateImageReader2(image_path.string().c_str()));

    if (!reader) {
        cerr << "Error: No reader found for: " << image_path << endl;
        return nullptr;
    }

    reader->SetFileName(image_path.string().c_str());
    reader->Update();

    return reader->GetOutput();
}

/// Load multiple images as `vtkImageData`
inline vector<Ptr<vtkImageData>> gather_vtk_images(const vector<path>& image_paths) {
    vector<Ptr<vtkImageData>> images;

    for (const auto& img_path : image_paths) {
        cout << "--------------------------------\n"
             << "Processing image: " << img_path.filename() << endl;

        if (!exists(img_path)) {
            cerr << "  Warning: Image not found: " << img_path << endl;
            continue;
        }

        // Load image using VTK
        auto vtk_image = load_vtk_image(img_path);
        if (!vtk_image) {
            cerr << "  Error: Failed to load VTK image: " << img_path << endl;
            continue;
        }

        cout << "  Loaded VTK image: (" << vtk_image->GetDimensions()[0] << "x"
             << vtk_image->GetDimensions()[1]
             << ", channels=" << vtk_image->GetNumberOfScalarComponents() << ")" << endl;

        images.push_back(vtk_image);
    }

    return images;
}


inline void save_vtk_image(const Ptr<vtkImageData>& vtk_image, const path& save_path) {
    if (!vtk_image) {
        cerr << "Error: Cannot save null vtkImageData to: " << save_path << endl;
        return;
    }

    // Check if the path has .png extension
    if (save_path.extension() != ".png") {
        cerr << "Error: Save path must have .png extension: " << save_path << endl;
        return;
    }

    auto writer = Ptr<vtkPNGWriter>::New();
    writer->SetFileName(save_path.string().c_str());
    writer->SetInputData(vtk_image);
    writer->Write();
}


#endif  // ENABLE_VTK
