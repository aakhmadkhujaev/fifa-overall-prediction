from typing import Any, cast

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    PreprocessorProtocol,
)


def make_split_loaders():
    features = torch.zeros(2, 3)
    targets = torch.ones(2, 1)
    return {
        name: DataLoader(TensorDataset(features, targets), batch_size=2)
        for name in ("train", "val", "test")
    }


def test_configuration_defaults_and_validation():
    config = ExperimentConfig("smoke", "goalkeeper")

    assert config.device == "cpu"
    assert config.num_workers == 0
    assert (config.hidden_size1, config.hidden_size2) == (64, 32)

    with pytest.raises(ValueError, match="population"):
        ExperimentConfig("smoke", cast(Any, "all"))
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig("smoke", "outfield", learning_rate=0)
    with pytest.raises(ValueError, match="batch_size"):
        ExperimentConfig("smoke", "outfield", batch_size=0)
    with pytest.raises(ValueError, match="epochs"):
        ExperimentConfig("smoke", "outfield", epochs=cast(Any, 1.5))
    with pytest.raises(ValueError, match="device"):
        ExperimentConfig("smoke", "outfield", device="not-a-device")


def test_configuration_is_frozen():
    config = ExperimentConfig("frozen", "outfield")

    with pytest.raises(AttributeError):
        setattr(config, "epochs", 1)


def test_result_captures_configuration_and_metrics():
    config = ExperimentConfig("result", "outfield")
    result = ExperimentResult("result", "outfield", config, 2, 1.0, 1.0, 1.5, 0.5, 2.0, 2.5, 0.1, 3.0)

    assert result.configuration is config
    assert result.best_epoch == 2
    assert result.test_r2 == 0.1


@pytest.mark.parametrize("population", ["goalkeeper", "outfield"])
def test_runner_propagates_configuration_and_keeps_test_out_of_training(monkeypatch, population):
    config = ExperimentConfig(
        "runner-test", population, batch_size=4, learning_rate=0.02,
        epochs=5, patience=2, random_seed=19, hidden_size1=11, hidden_size2=7,
    )
    loaders = make_split_loaders()
    model_calls = []
    train_calls = []
    events = []

    class FakeModel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2):
            super().__init__()
            model_calls.append((input_size, hidden_size1, hidden_size2))
            self.bias = nn.Parameter(torch.ones(1))

        def forward(self, features):
            return self.bias.expand(features.shape[0], 1)

    monkeypatch.setattr(
        "src.experiments.experiment.select_features",
        lambda data, is_goalkeeper: data,
    )

    def fake_preprocessor(is_goalkeeper: bool) -> PreprocessorProtocol:
        assert is_goalkeeper is (population == "goalkeeper")

        class FakePreprocessor:
            def process(
                self, data: pd.DataFrame
            ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, Any]]:
                return ({"train": pd.DataFrame([[1, 2, 3]]), "val": pd.DataFrame([[1, 2, 3]]), "test": pd.DataFrame([[1, 2, 3]])}, {}, {})

        return FakePreprocessor()

    def fake_dataloaders(features, targets, batch_size, num_workers):
        assert batch_size == 4
        assert num_workers == 0
        return loaders

    def fake_train(model, train_loader, val_loader, **kwargs):
        assert train_loader is loaders["train"]
        assert val_loader is loaders["val"]
        assert val_loader is not loaders["test"]
        assert kwargs == {
            "learning_rate": 0.02, "epochs": 5, "device": "cpu", "patience": 2,
            "seed": 19, "checkpoint_path": None,
        }
        train_calls.append((train_loader, val_loader))
        events.append("train")
        return {"best_epoch": 1, "best_val_mae": 1.0, "val_mae": [1.0], "val_rmse": [1.5], "val_r2": [0.5]}

    runner = ExperimentRunner(
        config,
        preprocessor_factory=fake_preprocessor,
        dataloader_factory=fake_dataloaders,
        model_factory=FakeModel,
        train_function=fake_train,
    )
    result = runner.run(pd.DataFrame({"overall": [1]}))

    events.append("test")
    assert model_calls == [(3, 11, 7)]
    assert len(train_calls) == 1
    assert result.population == population
    assert result.validation_mae == 1.0
    assert result.test_mae == 0.0


def test_runner_evaluates_test_after_training(monkeypatch):
    events = []
    config = ExperimentConfig("order", "outfield")
    monkeypatch.setattr(
        "src.experiments.experiment.select_features",
        lambda data, is_goalkeeper: data,
    )

    def fake_train(model, train_loader, val_loader, **kwargs):
        events.append("train")
        return {"best_epoch": 1, "best_val_mae": 0.0, "val_mae": [0.0], "val_rmse": [0.0], "val_r2": [1.0]}

    class EventModel(nn.Module):
        def forward(self, features):
            events.append("test")
            return torch.ones(features.shape[0], 1)

    class PreprocessorStub:
        def process(self, data):
            return (
                {"train": pd.DataFrame([[1]]), "val": pd.DataFrame([[1]]), "test": pd.DataFrame([[1]])},
                {},
                {},
            )

    runner = ExperimentRunner(
        config,
        preprocessor_factory=lambda **kwargs: PreprocessorStub(),
        dataloader_factory=lambda *args, **kwargs: make_split_loaders(),
        model_factory=lambda **kwargs: EventModel(),
        train_function=fake_train,
    )
    runner.run(pd.DataFrame({"overall": [1]}))

    assert events[0] == "train"
    assert "test" in events[1:]


def test_runner_can_defer_test_evaluation(monkeypatch):
    config = ExperimentConfig("deferred", "outfield")
    monkeypatch.setattr(
        "src.experiments.experiment.select_features",
        lambda data, is_goalkeeper: data,
    )

    class FailingTestLoader:
        def __iter__(self):
            raise AssertionError("deferred execution must not access test data")

    class PreprocessorStub:
        def process(self, data):
            frame = pd.DataFrame([[1]])
            return ({"train": frame, "val": frame, "test": frame}, {}, {})

    def fake_dataloaders(*args, **kwargs):
        loaders = make_split_loaders()
        loaders["test"] = FailingTestLoader()
        return loaders

    def fake_train(model, train_loader, val_loader, **kwargs):
        return {
            "best_epoch": 1,
            "best_val_mae": 0.5,
            "val_mae": [0.5],
            "val_rmse": [0.6],
            "val_r2": [0.7],
        }

    runner = ExperimentRunner(
        config,
        preprocessor_factory=lambda **kwargs: PreprocessorStub(),
        dataloader_factory=fake_dataloaders,
        model_factory=lambda **kwargs: nn.Linear(3, 1),
        train_function=fake_train,
    )

    result = runner.run(pd.DataFrame({"overall": [1]}), evaluate_test=False)

    assert result.validation_mae == 0.5
    assert result.validation_rmse == 0.6
    assert result.validation_r2 == 0.7
    assert result.test_mae is None
    assert result.test_rmse is None
    assert result.test_r2 is None