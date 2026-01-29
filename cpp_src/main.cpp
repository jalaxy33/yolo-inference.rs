/// Examples for calling Rust API from C++
#include <filesystem>
#include <iostream>

#include "cpp_ffi.rs.h"

using namespace std;
using namespace filesystem;

enum Experiment {
    OneImage,
    SmallBatch,
    LargeBatch,
    UnbatchableModel,
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
        default:
            throw runtime_error("Unknown experiment");
    }
}

int main() {
    path project_root = PROJECT_ROOT;
    path config_dir = project_root / "assets/configs";
    assert_path_exists(project_root);

    Experiment experiment = LargeBatch;
    path config_toml = get_config_path(config_dir, experiment);
    assert_path_exists(config_toml);

    cout << "Using config: " << config_toml << endl;

    // Run prediction
    predict_from_toml(config_toml);
    return 0;
}