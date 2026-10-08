import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import build_prediction_frame
from src.evaluation.v1_validation import find_absolute_paths, main, validate_v1_artifacts
from src.features.feature_engineering import get_goalkeeper_features, get_outfield_features

EPOCHS = {"goalkeeper": 5, "outfield": 6}
BEST = {"goalkeeper": 2, "outfield": 6}
STOPPED = {"goalkeeper": True, "outfield": False}
ROWS = {"goalkeeper": 4, "outfield": 7}


def make_evaluation():
    populations = {}
    for population, features in (
        ("goalkeeper", get_goalkeeper_features()),
        ("outfield", get_outfield_features()),
    ):
        epochs = EPOCHS[population]
        mae = [1.0 / (step + 1) for step in range(epochs)]
        populations[population] = {
            "checkpoint_path": f"models/fifa_overall_{population}_v1.pt",
            "metrics": {"mae": 0.4, "rmse": 0.5, "r2": 0.99, "test_samples": ROWS[population]},
            "training_history": {
                "train_loss": [1.0] * epochs,
                "validation_loss": [1.0] * epochs,
                "validation_mae": mae,
                "validation_rmse": [1.0] * epochs,
                "validation_r2": [0.5] * epochs,
                "best_epoch": BEST[population],
                "best_validation_mae": mae[BEST[population] - 1],
                "epochs_completed": epochs,
                "stopped_early": STOPPED[population],
            },
            "checkpoint_metadata": {
                "feature_names": list(features),
                "input_feature_count": len(features),
            },
        }
    return {
        "evaluation": {"dataset_path": "data/raw/male_players (legacy).csv"},
        "populations": populations,
    }


def make_record():
    record: dict[str, Any] = {"dataset": {"path": "data/raw/male_players (legacy).csv"}}
    for population in ("goalkeeper", "outfield"):
        mae = [1.0 / (step + 1) for step in range(EPOCHS[population])]
        record[population] = {
            "checkpoint_path": f"models/fifa_overall_{population}_v1.pt",
            "best_epoch": BEST[population],
            "best_validation_mae": mae[BEST[population] - 1],
            "epochs_completed": EPOCHS[population],
            "early_stopping": {"stopped_early": STOPPED[population], "patience": 7},
            "test": {"mae": 0.4, "rmse": 0.5, "r2": 0.99},
        }
    return record


def make_predictions():
    frames = []
    start = 0
    for population in ("goalkeeper", "outfield"):
        rows = ROWS[population]
        actual = 60.0 + np.arange(rows)
        frames.append(
            build_prediction_frame(
                list(range(start, start + rows)), actual, actual + 0.5, population
            )
        )
        start += rows
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def paths(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    predictions = tmp_path / "predictions.csv"
    record = tmp_path / "record.json"
    evaluation.write_text(json.dumps(make_evaluation()), encoding="utf-8")
    record.write_text(json.dumps(make_record()), encoding="utf-8")
    make_predictions().to_csv(predictions, index=False)
    return evaluation, predictions, record


def problems_for(paths, **kwargs):
    return validate_v1_artifacts(*paths, **kwargs)


def rewrite(path, mutate):
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_valid_artifacts_have_no_problems(paths):
    assert problems_for(paths) == []


def test_reference_record_with_identical_metrics_passes(paths):
    assert problems_for(paths, reference_record_path=paths[2]) == []


@pytest.mark.parametrize(
    "value",
    ["C:\\Users\\someone\\data.csv", "D:/data/x.csv", "/home/user/data.csv", "\\\\server\\share\\x"],
)
def test_absolute_paths_are_flagged(value):
    assert find_absolute_paths({"a": {"b": [value]}}, "doc") == [
        "doc: absolute path at $.a.b[0]"
    ]


@pytest.mark.parametrize(
    "value", ["data/raw/male_players (legacy).csv", "models/a.pt", "<60", "60-69", "90+", "outfield"]
)
def test_relative_paths_and_labels_are_not_flagged(value):
    assert find_absolute_paths({"a": value}, "doc") == []


def test_absolute_path_in_evaluation_is_reported(paths):
    rewrite(paths[0], lambda d: d["evaluation"].update(dataset_path="C:\\private\\x.csv"))
    assert any("absolute path" in problem for problem in problems_for(paths))


def test_absolute_path_in_training_record_is_reported(paths):
    rewrite(paths[2], lambda d: d["goalkeeper"].update(checkpoint_path="/home/me/m.pt"))
    assert any("training record JSON: absolute path" in p for p in problems_for(paths))


def test_truncated_history_is_reported_against_training_record(paths):
    """The pre-fix defect: history ends at the best epoch and stopped_early is False."""

    def truncate(document):
        history = document["populations"]["goalkeeper"]["training_history"]
        for key in ("train_loss", "validation_loss", "validation_mae", "validation_rmse", "validation_r2"):
            history[key] = history[key][: BEST["goalkeeper"]]
        history["epochs_completed"] = BEST["goalkeeper"]
        history["stopped_early"] = False

    rewrite(paths[0], truncate)
    problems = problems_for(paths)
    assert any("goalkeeper: epochs_completed is 2" in problem for problem in problems)
    assert any("goalkeeper: stopped_early is False" in problem for problem in problems)


def test_history_length_mismatch_is_reported(paths):
    rewrite(
        paths[0],
        lambda d: d["populations"]["outfield"]["training_history"]["validation_mae"].pop(),
    )
    assert any("history lengths" in problem for problem in problems_for(paths))


def test_best_epoch_out_of_range_is_reported(paths):
    rewrite(
        paths[0],
        lambda d: d["populations"]["outfield"]["training_history"].update(best_epoch=99),
    )
    assert any("best_epoch" in problem for problem in problems_for(paths))


def test_best_validation_mae_mismatch_is_reported(paths):
    rewrite(
        paths[0],
        lambda d: d["populations"]["outfield"]["training_history"].update(best_validation_mae=9.0),
    )
    assert any("best_validation_mae" in problem for problem in problems_for(paths))


def test_metric_disagreement_with_training_record_is_reported(paths):
    rewrite(paths[0], lambda d: d["populations"]["outfield"]["metrics"].update(mae=0.9))
    assert any("evaluated test mae differs" in problem for problem in problems_for(paths))


def test_metric_change_from_reference_is_reported(paths, tmp_path):
    reference = copy.deepcopy(make_record())
    reference["goalkeeper"]["test"]["rmse"] = 0.51
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    problems = problems_for(paths, reference_record_path=reference_path)
    assert any("goalkeeper: test rmse changed" in problem for problem in problems)


def test_row_count_mismatch_is_reported(paths):
    make_predictions().iloc[1:].to_csv(paths[1], index=False)
    assert any("prediction rows, expected" in problem for problem in problems_for(paths))


def test_player_id_overlap_between_populations_is_reported(paths):
    frame = make_predictions()
    frame.loc[frame["population"] == "outfield", "player_id"] = list(range(ROWS["outfield"]))
    frame.to_csv(paths[1], index=False)
    assert any("overlap" in problem for problem in problems_for(paths))


def test_repeated_player_ids_within_a_population_are_allowed(paths):
    frame = make_predictions()
    frame.loc[frame["population"] == "goalkeeper", "player_id"] = 1  # several records per player
    frame.to_csv(paths[1], index=False)
    assert problems_for(paths) == []


def test_wrong_prediction_columns_are_reported(paths):
    make_predictions().rename(columns={"error": "residual"}).to_csv(paths[1], index=False)
    assert any("columns" in problem for problem in problems_for(paths))


def test_inconsistent_error_columns_are_reported(paths):
    frame = make_predictions()
    frame.loc[0, "error"] = 5.0
    frame.to_csv(paths[1], index=False)
    assert any("error != predicted_overall" in problem for problem in problems_for(paths))


def test_non_finite_predictions_are_reported(paths):
    frame = make_predictions()
    frame.loc[0, "predicted_overall"] = np.nan
    frame.to_csv(paths[1], index=False)
    assert any("NaN or infinite" in problem for problem in problems_for(paths))


def test_feature_order_change_is_reported(paths):
    def swap(document):
        names = document["populations"]["goalkeeper"]["checkpoint_metadata"]["feature_names"]
        names[0], names[1] = names[1], names[0]

    rewrite(paths[0], swap)
    assert any("feature order" in problem for problem in problems_for(paths))


def test_feature_count_mismatch_is_reported(paths):
    rewrite(
        paths[0],
        lambda d: d["populations"]["outfield"]["checkpoint_metadata"].update(input_feature_count=3),
    )
    assert any("input_feature_count" in problem for problem in problems_for(paths))


def test_main_reports_success_and_failure(monkeypatch, paths, capsys):
    monkeypatch.setattr(
        "src.evaluation.v1_validation.validate_v1_artifacts", lambda **kwargs: []
    )
    assert main([]) == 0
    assert "consistent" in capsys.readouterr().out
    monkeypatch.setattr(
        "src.evaluation.v1_validation.validate_v1_artifacts", lambda **kwargs: ["broken"]
    )
    assert main([]) == 1
    assert "PROBLEM: broken" in capsys.readouterr().out
