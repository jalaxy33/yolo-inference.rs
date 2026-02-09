#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace std;
using namespace filesystem;

inline bool assert_path_exists(const path& p) {
    if (!exists(p)) {
        cerr << "Error: Path does not exist: " << p << endl;
        exit(1);
    }
    return true;
}

inline vector<path> list_image_paths(const path& image_dir) {
    assert_path_exists(image_dir);

    vector<path> image_paths;
    const vector<string> image_extensions = {"jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "tif"};

    for (const auto& entry : directory_iterator(image_dir)) {
        if (entry.is_regular_file()) {
            path file_path = entry.path();
            string ext = file_path.extension().string();

            // Remove leading dot and convert to lowercase for comparison
            if (!ext.empty() && ext[0] == '.') {
                ext = ext.substr(1);
            }
            for (char& c : ext) {
                c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
            }

            if (find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
                image_paths.push_back(file_path);
            }
        }
    }

    return image_paths;
}
