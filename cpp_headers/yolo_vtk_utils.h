#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "image_utils.h"
#include "yolo_common.h"

#ifdef ENABLE_VTK
#include "vtk_utils.h"

using namespace std;

// -- VTK-Rust Conversion utilities -------------------------------------

/// Convert `vtkImageData` to `RustImage`
inline Box<RustImage> vtk2rust(vtkImageData* vtk_image) {
    if (!vtk_image) {
        cerr << "Warning: vtk_image is null, returning empty RustImage" << endl;
        return yolo_inference::image_from_bytes(nullptr, 0, 0, 0);
    }

    int* dims = vtk_image->GetDimensions();
    int width = dims[0];
    int height = dims[1];
    int channels = vtk_image->GetNumberOfScalarComponents();

    const uint8_t* bytes = static_cast<const uint8_t*>(vtk_image->GetScalarPointer());

    // Copy to buffer and flip vertically to match Rust image coordinate system (top-left origin)
    size_t total_size = width * height * channels;
    vector<uint8_t> buffer(bytes, bytes + total_size);
    flip_vertical_inplace(buffer.data(), width, height, channels);

    return yolo_inference::image_from_bytes(buffer.data(), static_cast<uint32_t>(width),
                                            static_cast<uint32_t>(height),
                                            static_cast<uint32_t>(channels));
}

/// Convert `RustImage` to `vtkImageData`
inline Ptr<vtkImageData> rust2vtk(const RustImage& rs_image) {
    ImageInfo info = yolo_inference::get_image_info(rs_image);
    Vec<uint8_t> bytes = yolo_inference::image_to_bytes(rs_image);

    // Return nullptr if no image data
    if (bytes.empty()) {
        return nullptr;
    }

    // Flip bytes to match VTK coordinate system (bottom-left origin)
    flip_vertical_inplace(bytes.data(), info.width, info.height, info.channels);

    auto vtk_image = Ptr<vtkImageData>::New();
    vtk_image->SetDimensions(static_cast<int>(info.width), static_cast<int>(info.height), 1);
    vtk_image->AllocateScalars(VTK_UNSIGNED_CHAR, static_cast<int>(info.channels));

    std::memcpy(vtk_image->GetScalarPointer(), bytes.data(), bytes.size());
    return vtk_image;
}

/// Convert multiple `vtkImageData` to `RustImage`
inline Vec<Box<RustImage>> multi_vtk2rust(const vector<Ptr<vtkImageData>>& vtk_images) {
    Vec<Box<RustImage>> rs_images;
    for (const auto& vtk_img : vtk_images) {
        rs_images.push_back(vtk2rust(vtk_img));
    }
    return rs_images;
}

// -- Prediction results utilities -------------------------------------

inline Ptr<vtkImageData> get_annotated(const Box<InferResult>& result) {
    Box<RustImage> annotated = yolo_inference::get_result_annotated(*result);
    return rust2vtk(*annotated);
}

inline vector<Ptr<vtkImageData>> get_batch_annotated(const Vec<Box<InferResult>>& results) {
    vector<Ptr<vtkImageData>> annotated_images;
    for (const auto& res : results) {
        annotated_images.push_back(get_annotated(res));
    }
    return annotated_images;
}

#endif  // ENABLE_VTK
