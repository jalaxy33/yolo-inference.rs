#include <filesystem>
#include <iostream>

#include "path_utils.h"
#include "yolo-inference/src/ffi/cpp_ffi.rs.h"

using namespace std;
using namespace filesystem;

enum Experiment {
    OneImage,
    SmallBatch,
    LargeBatch,
    UnbatchableModel,
};

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
        default:
            throw runtime_error("Unknown experiment");
    }
}

int main() {
    path project_root = PROJECT_ROOT;
    path config_dir = project_root / "assets/configs";
    assert_path_exists(config_dir);

    Experiment experiment = SmallBatch;
    path config_toml = get_config_path(config_dir, experiment);
    assert_path_exists(config_toml);

    cout << "Using config: " << config_toml << endl;

    yolo_inference::predict_from_toml(config_toml);

    return 0;
}