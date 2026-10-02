import json

import numpy as np
import pandas as pd
import torch

from src.experiments import baseline_training
from src.features.feature_engineering import (
    get_goalkeeper_features,
    get_outfield_features,
)
from src.preprocessing.preprocessor import Preprocessor
from src.training.model import FIFAOverallModel


def make_synthetic_fifa_data() -> pd.DataFrame:
    features = set(get_goalkeeper_features() + get_outfield_features())
    rows = []
    for player_id in range(12):
        for record_number in range(3):
            row = {
                "player_id": player_id,
                "player_positions": "GK" if player_id < 4 else "ST",
                "overall": 60 + player_id + record_number,
            }
            for feature in features:
                row[feature] = (
                    "Right"
                    if feature == "preferred_foot" and record_number % 2 == 0
                    else "Left"
                    if feature == "preferred_foot"
                    else float(50 + player_id + record_number)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_baseline_orchestration_uses_independent_configured_models(
    monkeypatch, tmp_path
):
    synthetic_data = make_synthetic_fifa_data()
    training_calls = []

    monkeypatch.setattr(baseline_training, "load_dataset", lambda path: synthetic_data)

    def fake_train_model(model, train_loader, val_loader, **kwargs):
        training_calls.append((model, train_loader, val_loader, kwargs))
        checkpoint_path = kwargs["checkpoint_path"]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"synthetic checkpoint")
        return {
            "val_rmse": [2.0],
            "val_r2": [0.25],
            "best_epoch": 1,
            "best_val_mae": 1.5,
        }

    monkeypatch.setattr(baseline_training, "train_model", fake_train_model)

    results_path = tmp_path / "baseline_training_results.json"
    results = baseline_training.run_baseline_experiment(
        dataset_path=tmp_path / "synthetic.csv",
        models_dir=tmp_path / "models",
        results_path=results_path,
    )

    assert len(training_calls) == 2
    expected_input_sizes = {
        len(get_goalkeeper_features()),
        len(get_outfield_features()),
    }
    assert {call[0].input_size for call in training_calls} == expected_input_sizes

    for model, train_loader, val_loader, kwargs in training_calls:
        assert train_loader.batch_size == 256
        assert val_loader.batch_size == 256
        assert train_loader.num_workers == 0
        assert val_loader.num_workers == 0
        assert kwargs["learning_rate"] == 0.001
        assert kwargs["epochs"] == 30
        assert kwargs["patience"] == 7
        assert kwargs["device"] == "cpu"
        assert kwargs["seed"] == 42

    assert results["configuration"] == {
        "random_seed": 42,
        "batch_size": 256,
        "learning_rate": 0.001,
        "epochs": 30,
        "patience": 7,
        "device": "cpu",
        "hidden_sizes": [64, 32],
        "num_workers": 0,
    }
    assert results["goalkeeper"]["model_type"] == "goalkeeper"
    assert results["outfield"]["model_type"] == "outfield"
    assert results["goalkeeper"]["best_epoch"] == 1
    assert results["outfield"]["best_validation_rmse"] == 2.0
    assert np.isfinite(results["goalkeeper"]["test_mae"])
    assert np.isfinite(results["outfield"]["test_rmse"])
    assert results_path.exists()
    assert json.loads(results_path.read_text(encoding="utf-8"))["dataset"]["rows"] == 36


def test_baseline_uses_grouped_split_without_player_overlap(monkeypatch, tmp_path):
    synthetic_data = make_synthetic_fifa_data()
    observed_splits = []
    original_split = Preprocessor._split_data

    def recording_split(preprocessor, df, group_col, test_size, val_size, random_state=42):
        split = original_split(
            preprocessor, df, group_col, test_size, val_size, random_state
        )
        observed_splits.append(
            [set(frame[group_col]) for frame in split]
        )
        return split

    def fake_train_model(model, train_loader, val_loader, **kwargs):
        return {
            "val_rmse": [2.0],
            "val_r2": [0.25],
            "best_epoch": 1,
            "best_val_mae": 1.5,
        }

    monkeypatch.setattr(baseline_training, "load_dataset", lambda path: synthetic_data)
    monkeypatch.setattr(Preprocessor, "_split_data", recording_split)
    monkeypatch.setattr(baseline_training, "train_model", fake_train_model)

    baseline_training.run_baseline_experiment(
        dataset_path=tmp_path / "synthetic.csv",
        models_dir=tmp_path / "models",
        results_path=tmp_path / "results.json",
    )

    assert len(observed_splits) == 2
    for train_ids, validation_ids, test_ids in observed_splits:
        assert train_ids.isdisjoint(validation_ids)
        assert train_ids.isdisjoint(test_ids)
        assert validation_ids.isdisjoint(test_ids)


def test_baseline_uses_separate_gk_and_outfield_preprocessors(monkeypatch, tmp_path):
    synthetic_data = make_synthetic_fifa_data()
    preprocessing_modes = []
    original_preprocessor = baseline_training.Preprocessor

    class RecordingPreprocessor(original_preprocessor):
        def __init__(self, is_goalkeeper=False):
            preprocessing_modes.append(is_goalkeeper)
            super().__init__(is_goalkeeper=is_goalkeeper)

    def fake_train_model(model, train_loader, val_loader, **kwargs):
        return {
            "val_rmse": [2.0],
            "val_r2": [0.25],
            "best_epoch": 1,
            "best_val_mae": 1.5,
        }

    monkeypatch.setattr(baseline_training, "load_dataset", lambda path: synthetic_data)
    monkeypatch.setattr(baseline_training, "Preprocessor", RecordingPreprocessor)
    monkeypatch.setattr(baseline_training, "train_model", fake_train_model)

    baseline_training.run_baseline_experiment(
        dataset_path=tmp_path / "synthetic.csv",
        models_dir=tmp_path / "models",
        results_path=tmp_path / "results.json",
    )

    assert preprocessing_modes == [True, False]


def test_test_evaluation_runs_after_training_and_is_not_validation_loader(
    monkeypatch, tmp_path
):
    synthetic_data = make_synthetic_fifa_data()
    events = []
    observed_loaders = []
    original_create_dataloaders = baseline_training.create_dataloaders
    original_evaluate_test_set = baseline_training._evaluate_test_set

    def recording_create_dataloaders(*args, **kwargs):
        dataloaders = original_create_dataloaders(*args, **kwargs)
        observed_loaders.append(dataloaders)
        return dataloaders

    def fake_train_model(model, train_loader, val_loader, **kwargs):
        current_loaders = observed_loaders[-1]
        assert train_loader is current_loaders["train"]
        assert val_loader is current_loaders["val"]
        assert val_loader is not current_loaders["test"]
        events.append("train_model_complete")
        return {
            "val_rmse": [2.0],
            "val_r2": [0.25],
            "best_epoch": 1,
            "best_val_mae": 1.5,
        }

    def recording_evaluate_test_set(model, test_loader):
        events.append("test_evaluation")
        return original_evaluate_test_set(model, test_loader)

    monkeypatch.setattr(baseline_training, "load_dataset", lambda path: synthetic_data)
    monkeypatch.setattr(
        baseline_training, "create_dataloaders", recording_create_dataloaders
    )
    monkeypatch.setattr(baseline_training, "train_model", fake_train_model)
    monkeypatch.setattr(
        baseline_training, "_evaluate_test_set", recording_evaluate_test_set
    )

    baseline_training.run_baseline_experiment(
        dataset_path=tmp_path / "synthetic.csv",
        models_dir=tmp_path / "models",
        results_path=tmp_path / "results.json",
    )

    assert events == [
        "train_model_complete",
        "test_evaluation",
        "train_model_complete",
        "test_evaluation",
    ]


def test_baseline_checkpoints_load_into_compatible_models(monkeypatch, tmp_path):
    synthetic_data = make_synthetic_fifa_data()
    monkeypatch.setattr(baseline_training, "load_dataset", lambda path: synthetic_data)
    monkeypatch.setattr(baseline_training, "EPOCHS", 1)
    monkeypatch.setattr(baseline_training, "PATIENCE", 1)

    models_dir = tmp_path / "models"
    results = baseline_training.run_baseline_experiment(
        dataset_path=tmp_path / "synthetic.csv",
        models_dir=models_dir,
        results_path=tmp_path / "results.json",
    )

    for model_type in ("goalkeeper", "outfield"):
        checkpoint_path = models_dir / f"fifa_overall_{model_type}_baseline.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "training_config" in checkpoint
        assert "best_epoch" in checkpoint
        assert "best_val_mae" in checkpoint

        model = FIFAOverallModel(input_size=results[model_type]["input_features"])
        model.load_state_dict(checkpoint["model_state_dict"])
