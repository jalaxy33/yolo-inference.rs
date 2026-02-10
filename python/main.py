import os
from pathlib import Path
from enum import Enum, unique
from yolo_inference import predict_from_toml


@unique
class Experiment(Enum):
    OneImage = 0
    SmallBatch = 1
    LargeBatch = 2
    UnbatchableModel = 3


def main() -> None:
    os.chdir(Path(__file__).parent)

    project_root = Path("..").resolve()
    config_dir = project_root.joinpath("assets/configs")
    assert config_dir.is_dir(), f"Config directory does not exist: {config_dir}"

    experiment = Experiment.LargeBatch
    match experiment:
        case Experiment.OneImage:
            config_toml = config_dir.joinpath("one-image.toml")
        case Experiment.SmallBatch:
            config_toml = config_dir.joinpath("small-batch.toml")
        case Experiment.LargeBatch:
            config_toml = config_dir.joinpath("large-batch.toml")
        case Experiment.UnbatchableModel:
            config_toml = config_dir.joinpath("unbatchable-model.toml")
        case _:
            raise ValueError(f"Unknown experiment: {experiment}")

    assert config_toml.is_file() and config_toml.suffix == ".toml", (
        f"Invalid config toml: {config_toml}"
    )
    print(f"Using config: {config_toml}")

    # Run prediction
    predict_from_toml(str(config_toml), str(project_root))


if __name__ == "__main__":
    main()
