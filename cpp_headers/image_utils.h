#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

using namespace std;

/// Flip image data vertically (along Y axis) - out of place version
/// @param dst destination buffer
/// @param src source buffer
/// @param width image width in pixels
/// @param height image height in pixels
/// @param channels number of channels per pixel
inline void flip_vertical(uint8_t* dst, const uint8_t* src, int width, int height, int channels) {
    size_t row_size = width * channels;
    for (int y = 0; y < height; ++y) {
        int src_row = height - 1 - y;  // read from bottom to top
        memcpy(dst + y * row_size, src + src_row * row_size, row_size);
    }
}

/// Flip image data vertically (along Y axis) - in-place version
/// @param image image buffer to flip in place
/// @param width image width in pixels
/// @param height image height in pixels
/// @param channels number of channels per pixel
inline void flip_vertical_inplace(uint8_t* image, int width, int height, int channels) {
    size_t row_size = width * channels;
    vector<uint8_t> temp(row_size);
    for (int top = 0, bottom = height - 1; top < bottom; ++top, --bottom) {
        uint8_t* row_top = image + top * row_size;
        uint8_t* row_bottom = image + bottom * row_size;
        memcpy(temp.data(), row_top, row_size);
        memcpy(row_top, row_bottom, row_size);
        memcpy(row_bottom, temp.data(), row_size);
    }
}
