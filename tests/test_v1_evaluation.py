import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.evaluation.error_analysis import (
    build_prediction_frame,
    calculate_error_direction,
    largest_absolute_errors,
    summarize_rating_ranges,
)
from src.evaluation import v1_evaluation
from src.evaluation.v1_evaluation import EvaluationError, evaluate_final_v1, evaluate_population
from src.experiments.artifact import V1Artifact
from src.features.feature_engineering import get_goalkeeper_features, get_outfield_features
from src.preprocessing.preprocessor import (
    Preprocessor,
    deterministic_group_split,
    split_with_target_handling,
)


def make_population_data(features, population, groups=20, records=2):
    rows = []
    for player_id in range(groups):
        for record in range(records):
            row = {
                "player_id": player_id,
                "player_positions": "GK" if population == "goalkeeper" else "ST",
                "overall": float(60 + player_id + record),
            }
            row.update({
                feature: ("Right" if feature == "preferred_foot" else float(40 + player_id + record))
                for feature in features
            })
            rows.append(row)
    return pd.DataFrame(rows)


class StubPreprocessor:
    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.target_col = "overall"
        self.group_col = "player_id"
        self.transform_calls = 0

    def transform_features(self, frame):
        self.transform_calls += 1
        transformed = frame.loc[:, self.feature_names].copy()
        if "preferred_foot" in transformed:
            transformed["preferred_foot"] = transformed["preferred_foot"].map({"Right": 1.0, "Left": 0.0})
        return transformed.astype(np.float32)


def make_artifact(feature_names, population):
    preprocessor = StubPreprocessor(feature_names)
    model = nn.Linear(len(feature_names), 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(70.0)
    model.eval()
    metadata = {
        "checkpoint_schema_version": 2,
        "population": population,
        "model_class": "FIFAOverallModel",
        "experiment_config": {"population": population},
        "input_features": len(feature_names),
        "input_feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "target_col": "overall",
        "group_col": "player_id",
        "preprocessing_schema_version": 1,
        "preprocessing_state": {"schema_version": 1},
        "model_config": {"model_class": "FIFAOverallModel"},
    }
    checkpoint = {
        "checkpoint_metadata": metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_epoch": 2,
        "best_val_mae": 1.5,
        "history": {
            "train_loss": [4.0, 2.0],
            "val_loss": [3.0, 1.0],
            "val_mae": [2.0, 1.5],
            "val_rmse": [2.1, 1.6],
            "val_r2": [0.8, 0.9],
            "epochs_completed": 2,
            "stopped_early": True,
        },
    }
    return V1Artifact(checkpoint=checkpoint, model=model, preprocessor=preprocessor)


def test_split_helper_matches_training_semantics_and_has_no_group_overlap():
    data = make_population_data(get_goalkeeper_features(), "goalkeeper")
    data.loc[0, "overall"] = np.nan
    expected = Preprocessor(is_goalkeeper=True)._split_data(
        data.dropna(subset=["overall"]), "player_id", 0.15, 0.15
    )
    actual = split_with_target_handling(data)
    for expected_frame, actual_frame in zip(expected, actual):
        assert expected_frame.index.tolist() == actual_frame.index.tolist()
    groups = [set(frame["player_id"]) for frame in actual]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_public_deterministic_split_is_repeatable():
    data = make_population_data(get_goalkeeper_features(), "goalkeeper")
    first = deterministic_group_split(data, "player_id", 0.15, 0.15)
    second = deterministic_group_split(data, "player_id", 0.15, 0.15)
    assert [frame.index.tolist() for frame in first] == [frame.index.tolist() for frame in second]


def test_prediction_frame_and_error_analysis():
    frame = build_prediction_frame([1, 2, 3], [60, 70, 80], [61, 68, 80], "goalkeeper")
    assert list(frame.columns) == ["player_id", "actual_overall", "predicted_overall", "error", "absolute_error", "population"]
    assert frame["error"].tolist() == [1.0, -2.0, 0.0]
    assert frame["absolute_error"].tolist() == [1.0, 2.0, 0.0]
    metrics = calculate_error_direction(frame["actual_overall"], frame["predicted_overall"])
    assert metrics["underprediction_count"] == 1
    assert metrics["overprediction_count"] == 1
    assert metrics["underprediction_rate"] == pytest.approx(1 / 3)
    assert metrics["overprediction_rate"] == pytest.approx(1 / 3)
    ranges = summarize_rating_ranges(frame)
    assert ranges["samples"].sum() == 3
    largest = largest_absolute_errors(frame, limit=2)
    assert largest["absolute_error"].tolist() == [2.0, 1.0]


def test_evaluate_population_uses_restored_preprocessor_without_fitting(monkeypatch):
    features = get_goalkeeper_features()
    data = make_population_data(features, "goalkeeper")
    artifact = make_artifact(features, "goalkeeper")

    def fail_process(*args, **kwargs):
        raise AssertionError("evaluation must not fit preprocessing")

    monkeypatch.setattr(Preprocessor, "process", fail_process)
    monkeypatch.setattr(v1_evaluation, "load_v1_checkpoint", lambda path: artifact)
    result, predictions = evaluate_population(
        data, population="goalkeeper", checkpoint_path="goalkeeper.pt", batch_size=3
    )

    assert artifact.preprocessor.transform_calls == 1
    assert result.split_sizes["test_size"] == len(predictions)
    assert result.metrics["test_samples"] == len(predictions)
    assert result.training_history["best_epoch"] == 2
    assert result.training_history["stopped_early"] is True
    assert predictions["population"].eq("goalkeeper").all()


def test_evaluation_rejects_incompatible_checkpoint_features_before_transform(monkeypatch):
    features = get_goalkeeper_features()
    data = make_population_data(features, "goalkeeper")
    artifact = make_artifact(features[:-1] + ["unexpected_feature"], "goalkeeper")

    def fail_process(*args, **kwargs):
        raise AssertionError("evaluation must not fit preprocessing")

    monkeypatch.setattr(Preprocessor, "process", fail_process)
    monkeypatch.setattr(v1_evaluation, "load_v1_checkpoint", lambda path: artifact)
    with pytest.raises(EvaluationError, match="feature order"):
        evaluate_population(data, population="goalkeeper", checkpoint_path="invalid.pt")
    assert artifact.preprocessor.transform_calls == 0


def test_evaluate_final_v1_writes_comparison_and_prediction_csv(monkeypatch, tmp_path):
    goalkeeper = make_population_data(get_goalkeeper_features(), "goalkeeper")
    outfield = make_population_data(get_outfield_features(), "outfield")
    dataset = pd.concat([goalkeeper, outfield], ignore_index=True)
    artifacts = {
        "goalkeeper": make_artifact(get_goalkeeper_features(), "goalkeeper"),
        "outfield": make_artifact(get_outfield_features(), "outfield"),
    }

    monkeypatch.setattr(v1_evaluation, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(v1_evaluation, "validate_dataset", lambda frame: {"target_exists": True})
    monkeypatch.setattr(v1_evaluation, "load_v1_checkpoint", lambda path: artifacts["goalkeeper" if "goalkeeper" in str(path) else "outfield"])
    results_path = tmp_path / "evaluation.json"
    predictions_path = tmp_path / "predictions.csv"
    output = evaluate_final_v1(results_path=results_path, predictions_path=predictions_path)

    assert set(output["populations"]) == {"goalkeeper", "outfield"}
    assert set(output["comparison"]) == {"goalkeeper", "outfield"}
    predictions = pd.read_csv(predictions_path)
    assert list(predictions.columns) == ["player_id", "actual_overall", "predicted_overall", "error", "absolute_error", "population"]
    assert len(predictions) == sum(result["metrics"]["test_samples"] for result in output["populations"].values())
    assert json.loads(results_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert np.isfinite(predictions[["actual_overall", "predicted_overall", "error", "absolute_error"]].to_numpy()).all()
