import pandas as pd
import pytest
import torch
from torch import nn

from src.experiments import final_v1
from src.experiments.artifact import load_v1_checkpoint, portable_path
from src.experiments.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner
from src.preprocessing.preprocessor import Preprocessor
from src.training.model import FIFAOverallModel


def test_final_configurations_are_fixed():
    configs = final_v1.final_v1_configs()
    assert len(configs) == 2
    assert [config.population for config in configs] == ["goalkeeper", "outfield"]
    assert configs[0].learning_rate == 0.01
    assert configs[0].batch_size == 128
    assert configs[1].learning_rate == 0.0017320508075688787
    assert configs[1].batch_size == 256
    assert all((config.epochs, config.patience, config.random_seed) == (30, 7, 42) for config in configs)
    assert all((config.hidden_size1, config.hidden_size2) == (64, 32) for config in configs)


def test_final_orchestration_runs_once_per_population(monkeypatch, tmp_path):
    dataset = pd.DataFrame({"player_positions": ["GK", "ST"], "overall": [80, 70]})
    calls = []

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self, population_data, *, checkpoint_path):
            calls.append((self.config.population, checkpoint_path.name, len(population_data)))
            return ExperimentResult(
                self.config.experiment_name, self.config.population, self.config,
                1, 1.0, 1.0, 1.5, 0.2, 2.0, 2.5, 0.1, 0.01,
                checkpoint_path=str(checkpoint_path), input_features=2,
                feature_names=("a", "b"), split_sizes={"train_size": 1, "val_size": 1, "test_size": 1},
                epochs_completed=1, stopped_early=False,
            )

    monkeypatch.setattr(final_v1, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(final_v1, "ExperimentRunner", FakeRunner)
    output = final_v1.run_final_v1_experiment(models_dir=tmp_path / "models", results_path=tmp_path / "results.json")

    assert calls == [("goalkeeper", "fifa_overall_goalkeeper_v1.pt", 1), ("outfield", "fifa_overall_outfield_v1.pt", 1)]
    assert output["goalkeeper"]["validation"]["mae"] == 1.0
    assert output["outfield"]["test"]["rmse"] == 2.5


def test_final_results_contain_no_machine_specific_paths(monkeypatch, tmp_path):
    dataset = pd.DataFrame({"player_positions": ["GK", "ST"], "overall": [80, 70]})

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self, population_data, *, checkpoint_path):
            return ExperimentResult(
                self.config.experiment_name, self.config.population, self.config,
                1, 1.0, 1.0, 1.5, 0.2, 2.0, 2.5, 0.1, 0.01,
                input_features=2, feature_names=("a", "b"),
                split_sizes={"train_size": 1, "val_size": 1, "test_size": 1},
                epochs_completed=1, stopped_early=False,
            )

    monkeypatch.setattr(final_v1, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(final_v1, "ExperimentRunner", FakeRunner)
    results_path = tmp_path / "results.json"
    output = final_v1.run_final_v1_experiment(
        models_dir=tmp_path / "models", results_path=results_path
    )

    assert output["dataset"]["path"] == "data/raw/male_players (legacy).csv"
    assert output["goalkeeper"]["checkpoint_path"] == "fifa_overall_goalkeeper_v1.pt"
    assert output["outfield"]["checkpoint_path"] == "fifa_overall_outfield_v1.pt"
    assert str(tmp_path) not in results_path.read_text(encoding="utf-8")


def test_runner_constructs_self_contained_checkpoint_metadata(monkeypatch, tmp_path):
    config = ExperimentConfig("metadata", "goalkeeper")
    captured = {}

    class PreprocessorStub:
        feature_names = ["age", "preferred_foot"]
        target_col = "overall"
        group_col = "player_id"

        def process(self, df, target_col="overall", group_col="player_id"):
            frame = pd.DataFrame([[1.0, 0.0]])
            return ({"train": frame, "val": frame, "test": frame}, {}, {"train_size": 1, "val_size": 1, "test_size": 1})

        def get_state(self):
            return {"schema_version": 1, "feature_names": self.feature_names}

    def fake_train(model, train_loader, val_loader, **kwargs):
        captured.update(kwargs)
        return {"best_epoch": 1, "best_val_mae": 1.0, "val_mae": [1.0], "val_rmse": [1.0], "val_r2": [0.0]}

    monkeypatch.setattr("src.experiments.experiment.select_features", lambda data, is_goalkeeper: data)
    runner = ExperimentRunner(
        config,
        preprocessor_factory=lambda **kwargs: PreprocessorStub(),
        dataloader_factory=lambda *args, **kwargs: {
            name: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(1, 2), torch.ones(1, 1)))
            for name in ("train", "val", "test")
        },
        model_factory=lambda **kwargs: nn.Linear(2, 1),
        train_function=fake_train,
    )
    runner.run(pd.DataFrame({"overall": [1]}), checkpoint_path=tmp_path / "v1.pt")

    metadata = captured["checkpoint_metadata"]
    assert metadata["checkpoint_schema_version"] == 2
    assert metadata["population"] == "goalkeeper"
    assert metadata["experiment_config"]["learning_rate"] == config.learning_rate
    assert metadata["feature_names"] == ["age", "preferred_foot"]
    assert metadata["preprocessor_state"] == {"schema_version": 1, "feature_names": ["age", "preferred_foot"]}


def test_v1_checkpoint_loads_and_reconstructs(tmp_path):
    preprocessor = Preprocessor(is_goalkeeper=True)
    preprocessor.feature_names = ["age", "preferred_foot"]
    preprocessor.numeric_features = ["age"]
    preprocessor.categorical_features = ["preferred_foot"]
    preprocessor.target_col = "overall"
    preprocessor.group_col = "player_id"
    preprocessor.num_imputer.fit([[20]])
    preprocessor.cat_imputer.fit([["Right"]])
    preprocessor.scaler.fit([[20, 1]])
    model = FIFAOverallModel(input_size=2)
    metadata = {
        "checkpoint_schema_version": 2,
        "population": "goalkeeper",
        "model_class": "FIFAOverallModel",
        "model_configuration": {"input_size": 2, "hidden_size1": 64, "hidden_size2": 32},
        "model_config": {"model_class": "FIFAOverallModel", "input_size": 2, "hidden_size1": 64, "hidden_size2": 32},
        "experiment_config": {"population": "goalkeeper", "learning_rate": 0.01},
        "input_features": 2,
        "input_feature_count": 2,
        "feature_names": ["age", "preferred_foot"],
        "target_col": "overall",
        "group_col": "player_id",
        "preprocessing_schema_version": 1,
        "preprocessing_state": preprocessor.get_state(),
    }
    path = tmp_path / "model.pt"
    torch.save({
        "checkpoint_metadata": metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_epoch": 1,
        "best_val_mae": 1.0,
    }, path)

    artifact = load_v1_checkpoint(path)
    transformed = artifact.preprocessor.transform_features(pd.DataFrame({"age": [25], "preferred_foot": ["Left"]}))
    with torch.no_grad():
        prediction = artifact.model(torch.tensor(transformed.to_numpy(), dtype=torch.float32))
    assert prediction.shape == (1, 1)
    assert list(artifact.preprocessor.feature_names) == ["age", "preferred_foot"]


def test_v1_checkpoint_validation_rejects_each_required_field(tmp_path):
    valid_path = tmp_path / "model.pt"
    test_v1_checkpoint_loads_and_reconstructs(tmp_path)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["checkpoint_metadata"]
    required_fields = (
        "checkpoint_schema_version", "population", "model_class", "experiment_config",
        "input_features", "input_feature_count", "feature_names", "target_col", "group_col",
        "preprocessing_schema_version", "preprocessing_state", "model_config",
    )
    for field in required_fields:
        invalid = {**checkpoint, "checkpoint_metadata": {key: value for key, value in metadata.items() if key != field}}
        path = tmp_path / f"missing_{field}.pt"
        torch.save(invalid, path)
        with pytest.raises(ValueError, match="missing"):
            load_v1_checkpoint(path)

    for field in ("model_state_dict", "optimizer_state_dict", "best_epoch", "best_val_mae"):
        invalid = {key: value for key, value in checkpoint.items() if key != field}
        path = tmp_path / f"missing_{field}.pt"
        torch.save(invalid, path)
        with pytest.raises(ValueError, match="missing"):
            load_v1_checkpoint(path)


def test_v1_checkpoint_validation_rejects_unsupported_model_class(tmp_path):
    preprocessor = Preprocessor(is_goalkeeper=True)
    preprocessor.feature_names = ["age"]
    preprocessor.numeric_features = ["age"]
    preprocessor.num_imputer.fit([[20]])
    preprocessor.scaler.fit([[20]])
    model = FIFAOverallModel(input_size=1)
    metadata = {
        "checkpoint_schema_version": 2, "population": "goalkeeper", "model_class": "OtherModel",
        "experiment_config": {"population": "goalkeeper"}, "input_features": 1,
        "input_feature_count": 1, "feature_names": ["age"], "target_col": "overall",
        "group_col": "player_id", "preprocessing_schema_version": 1,
        "preprocessing_state": preprocessor.get_state(),
        "model_config": {"model_class": "OtherModel", "input_size": 1, "hidden_size1": 64, "hidden_size2": 32},
    }
    path = tmp_path / "unsupported_model.pt"
    torch.save({"checkpoint_metadata": metadata, "model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "best_epoch": 1, "best_val_mae": 1.0}, path)
    with pytest.raises(ValueError, match="Unsupported model class"):
        load_v1_checkpoint(path)
