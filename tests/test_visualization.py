import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.evaluation.error_analysis import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    largest_absolute_errors,
    summarize_rating_ranges,
)
from src.visualization import (
    EvaluationData,
    VisualizationDataError,
    get_comparison,
    get_largest_errors,
    get_population_predictions,
    get_rating_ranges,
    get_training_history,
    load_evaluation_data,
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_largest_errors,
    plot_mae_by_rating_range,
    plot_population_comparison,
    plot_training_history,
)
from src.visualization import data as data_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_DIR = PROJECT_ROOT / "src" / "visualization"
GK_ROWS = 6
OF_ROWS = 40


def make_predictions() -> pd.DataFrame:
    frames = []
    for population, rows, offset in (("goalkeeper", GK_ROWS, 0), ("outfield", OF_ROWS, 1000)):
        actual = 55.0 + np.arange(rows) % 30
        error = np.where(np.arange(rows) % 2 == 0, 0.5, -0.8) + 0.01 * np.arange(rows)
        frames.append(
            build_prediction_frame(list(offset + np.arange(rows)), actual, actual + error, population)
        )
    return pd.concat(frames, ignore_index=True)


def make_history(epochs: int = 5, best_epoch: int = 4) -> dict:
    steps = np.arange(1, epochs + 1)
    return {
        "train_loss": list(10.0 / steps),
        "validation_loss": list(12.0 / steps),
        "validation_mae": list(3.0 / steps),
        "validation_rmse": list(4.0 / steps),
        "validation_r2": list(1 - 1.0 / (steps + 1)),
        "best_epoch": best_epoch,
        "best_validation_mae": 3.0 / best_epoch,
        "epochs_completed": epochs,
        "stopped_early": False,
    }


def make_results(predictions: pd.DataFrame) -> dict:
    populations = {}
    comparison = {}
    for index, population in enumerate(("goalkeeper", "outfield")):
        rows = pd.DataFrame(predictions.loc[predictions["population"] == population]).reset_index(drop=True)
        samples = len(rows)
        mae = float(np.mean(np.asarray(rows["absolute_error"], dtype=np.float64)))
        populations[population] = {
            "population": population,
            "checkpoint_path": "C:\\does\\not\\exist\\model.pt",
            "metrics": {"mae": mae, "test_samples": samples},
            "error_analysis": {
                "rating_ranges": summarize_rating_ranges(rows).to_dict(orient="records"),
                "largest_absolute_errors": largest_absolute_errors(rows, limit=5).to_dict(
                    orient="records"
                ),
            },
            "training_history": make_history(5 + index, 4 + index),
        }
        comparison[population] = {
            "mae": mae,
            "rmse": mae * 1.2,
            "r2": 0.97 + 0.01 * index,
            "test_samples": samples,
        }
    return {
        "schema_version": 1,
        "evaluation": {"dataset_path": "C:\\does\\not\\exist\\players.csv"},
        "populations": populations,
        "comparison": comparison,
    }


@pytest.fixture()
def predictions() -> pd.DataFrame:
    return make_predictions()


@pytest.fixture()
def results(predictions: pd.DataFrame) -> dict:
    return make_results(predictions)


@pytest.fixture()
def artifact_paths(tmp_path: Path, predictions: pd.DataFrame, results: dict) -> tuple[Path, Path]:
    results_path = tmp_path / "evaluation.json"
    predictions_path = tmp_path / "predictions.csv"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    return results_path, predictions_path


@pytest.fixture()
def evaluation_data(artifact_paths: tuple[Path, Path]) -> EvaluationData:
    return load_evaluation_data(*artifact_paths)


def offsets(collection: Any) -> np.ndarray:
    return np.asarray(collection.get_offsets(), dtype=np.float64)


def xdata(line: Any) -> np.ndarray:
    return np.asarray(line.get_xdata(), dtype=np.float64)


def ydata(line: Any) -> np.ndarray:
    return np.asarray(line.get_ydata(), dtype=np.float64)


def bar_heights(ax: Any) -> list[float]:
    return [float(cast(Rectangle, patch).get_height()) for patch in ax.patches]


def bar_widths(ax: Any) -> list[float]:
    return [float(cast(Rectangle, patch).get_width()) for patch in ax.patches]


def legend_texts(ax: Any) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def assert_figure(figure: Figure, axes_count: int = 1) -> None:
    assert isinstance(figure, Figure)
    assert len(figure.axes) == axes_count


# ---------------------------------------------------------------- loader


def test_load_valid_artifacts(artifact_paths, predictions):
    data = load_evaluation_data(*artifact_paths)

    assert isinstance(data, EvaluationData)
    assert list(data.predictions.columns) == list(PREDICTION_COLUMNS)
    assert len(data.predictions) == len(predictions)
    assert data.results["schema_version"] == 1


def test_evaluation_data_is_frozen(evaluation_data):
    with pytest.raises(AttributeError):
        evaluation_data.predictions = pd.DataFrame()  # type: ignore[misc]


def test_loader_does_not_use_paths_stored_in_json(artifact_paths):
    # The fixture JSON points at nonexistent Windows paths; loading must still succeed.
    assert load_evaluation_data(*artifact_paths) is not None


def test_default_paths_are_relative_to_project():
    assert data_module.DEFAULT_RESULTS_PATH == (
        PROJECT_ROOT / "reports" / "generated" / "v1_final_evaluation.json"
    )
    assert data_module.DEFAULT_PREDICTIONS_PATH == (
        PROJECT_ROOT / "reports" / "generated" / "v1_test_predictions.csv"
    )


def test_missing_json(artifact_paths, tmp_path):
    with pytest.raises(VisualizationDataError, match="JSON not found"):
        load_evaluation_data(tmp_path / "missing.json", artifact_paths[1])


def test_missing_csv(artifact_paths, tmp_path):
    with pytest.raises(VisualizationDataError, match="CSV not found"):
        load_evaluation_data(artifact_paths[0], tmp_path / "missing.csv")


def test_invalid_json_content(artifact_paths):
    artifact_paths[0].write_text("{not json", encoding="utf-8")
    with pytest.raises(VisualizationDataError, match="not valid JSON"):
        load_evaluation_data(*artifact_paths)


@pytest.mark.parametrize("version", [2, 0, "1", None, True])
def test_invalid_schema_version(artifact_paths, results, version):
    results["schema_version"] = version
    artifact_paths[0].write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(VisualizationDataError, match="schema_version"):
        load_evaluation_data(*artifact_paths)


def test_missing_population_section(artifact_paths, results):
    del results["populations"]["outfield"]
    artifact_paths[0].write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(VisualizationDataError, match="outfield"):
        load_evaluation_data(*artifact_paths)


def test_wrong_csv_columns(artifact_paths, predictions):
    predictions.rename(columns={"error": "residual"}).to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="columns"):
        load_evaluation_data(*artifact_paths)


def test_reordered_csv_columns_rejected(artifact_paths, predictions):
    predictions[list(reversed(PREDICTION_COLUMNS))].to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="columns"):
        load_evaluation_data(*artifact_paths)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_values(artifact_paths, predictions, bad_value):
    broken = predictions.copy()
    broken.loc[0, "predicted_overall"] = bad_value
    broken.to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="NaN or infinite"):
        load_evaluation_data(*artifact_paths)


def test_non_numeric_column(artifact_paths, predictions):
    broken = predictions.copy()
    broken["actual_overall"] = broken["actual_overall"].astype(str) + "x"
    broken.to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="numeric"):
        load_evaluation_data(*artifact_paths)


def test_invalid_population_value(artifact_paths, predictions):
    broken = predictions.copy()
    broken.loc[0, "population"] = "striker"
    broken.to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="striker"):
        load_evaluation_data(*artifact_paths)


def test_missing_population_value_rejected(artifact_paths, predictions):
    broken = predictions.copy()
    broken.loc[0, "population"] = np.nan
    broken.to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="population"):
        load_evaluation_data(*artifact_paths)


def test_population_row_count_mismatch(artifact_paths, predictions):
    predictions.iloc[1:].to_csv(artifact_paths[1], index=False)  # drops one goalkeeper row
    with pytest.raises(VisualizationDataError, match="test samples"):
        load_evaluation_data(*artifact_paths)


def test_empty_population(artifact_paths, predictions):
    predictions[predictions["population"] == "outfield"].to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="goalkeeper"):
        load_evaluation_data(*artifact_paths)


def test_csv_with_header_only(artifact_paths, predictions):
    predictions.iloc[0:0].to_csv(artifact_paths[1], index=False)
    with pytest.raises(VisualizationDataError, match="no rows"):
        load_evaluation_data(*artifact_paths)


def test_empty_csv_file(artifact_paths):
    artifact_paths[1].write_text("", encoding="utf-8")
    with pytest.raises(VisualizationDataError, match="could not be read"):
        load_evaluation_data(*artifact_paths)


def test_visualization_data_error_is_value_error():
    assert issubclass(VisualizationDataError, ValueError)


# ------------------------------------------------------------- accessors


def test_population_predictions_accessor(evaluation_data):
    rows = get_population_predictions(evaluation_data, "goalkeeper")
    assert len(rows) == GK_ROWS
    assert set(rows["population"]) == {"goalkeeper"}
    rows.loc[0, "actual_overall"] = -1.0
    assert (evaluation_data.predictions["actual_overall"] != -1.0).all()


def test_rating_ranges_accessor(evaluation_data):
    ranges = get_rating_ranges(evaluation_data, "outfield")
    assert {"rating_range", "samples", "mae"} <= set(ranges.columns)
    assert ranges["samples"].sum() == OF_ROWS


def test_training_history_accessor_returns_copy(evaluation_data):
    history = get_training_history(evaluation_data, "goalkeeper")
    history["train_loss"].append(99.0)
    assert 99.0 not in get_training_history(evaluation_data, "goalkeeper")["train_loss"]


def test_comparison_accessor_returns_copy(evaluation_data):
    comparison = get_comparison(evaluation_data)
    assert set(comparison) == {"goalkeeper", "outfield"}
    comparison["goalkeeper"]["mae"] = -1.0
    assert get_comparison(evaluation_data)["goalkeeper"]["mae"] != -1.0


def test_largest_errors_accessor(evaluation_data):
    errors = get_largest_errors(evaluation_data, "outfield")
    assert list(errors.columns) == list(PREDICTION_COLUMNS)
    assert len(errors) == 5


@pytest.mark.parametrize(
    "accessor",
    [
        get_population_predictions,
        get_rating_ranges,
        get_training_history,
        get_largest_errors,
    ],
)
def test_accessors_reject_unknown_population(evaluation_data, accessor):
    with pytest.raises(VisualizationDataError, match="Unknown population"):
        accessor(evaluation_data, "striker")


# --------------------------------------------------- actual vs predicted


def test_actual_vs_predicted_both_populations(predictions):
    figure = plot_actual_vs_predicted(predictions)

    assert_figure(figure)
    ax = figure.axes[0]
    assert ax.get_xlabel() == "Actual overall rating"
    assert ax.get_ylabel() == "Predicted overall rating"
    assert "Actual vs predicted" in ax.get_title()
    assert len(ax.collections) == 2
    assert sum(len(offsets(c)) for c in ax.collections) == len(predictions)
    reference = ax.lines[0]
    assert list(xdata(reference)) == list(ydata(reference))


def test_actual_vs_predicted_filters_population(predictions):
    figure = plot_actual_vs_predicted(predictions, "goalkeeper")

    ax = figure.axes[0]
    assert len(ax.collections) == 1
    points = offsets(ax.collections[0])
    gk = predictions[predictions["population"] == "goalkeeper"]
    assert len(points) == GK_ROWS
    assert np.allclose(points[:, 0], gk["actual_overall"])
    assert np.allclose(points[:, 1], gk["predicted_overall"])
    assert "goalkeeper" in ax.get_title()


def test_actual_vs_predicted_max_points_deterministic(predictions):
    def points(figure: Figure) -> np.ndarray:
        return np.vstack([offsets(c) for c in figure.axes[0].collections])

    first = plot_actual_vs_predicted(predictions, max_points=10)
    second = plot_actual_vs_predicted(predictions, max_points=10)

    assert len(points(first)) == 10
    assert np.array_equal(points(first), points(second))


@pytest.mark.parametrize("max_points", [1, 7, len(make_predictions()), 1000])
def test_actual_vs_predicted_max_points_never_exceeded(predictions, max_points):
    figure = plot_actual_vs_predicted(predictions, max_points=max_points)
    drawn = sum(len(offsets(c)) for c in figure.axes[0].collections)
    assert drawn == min(max_points, len(predictions))


@pytest.mark.parametrize("max_points", [0, -3, 2.5, True])
def test_actual_vs_predicted_invalid_max_points(predictions, max_points):
    with pytest.raises(VisualizationDataError, match="max_points"):
        plot_actual_vs_predicted(predictions, max_points=max_points)


def test_actual_vs_predicted_invalid_population(predictions):
    with pytest.raises(VisualizationDataError, match="Unknown population"):
        plot_actual_vs_predicted(predictions, "striker")


def test_actual_vs_predicted_missing_columns(predictions):
    with pytest.raises(VisualizationDataError, match="predicted_overall"):
        plot_actual_vs_predicted(predictions.drop(columns=["predicted_overall"]))


def test_actual_vs_predicted_empty_input(predictions):
    with pytest.raises(VisualizationDataError, match="No prediction rows"):
        plot_actual_vs_predicted(predictions.iloc[0:0])


def test_actual_vs_predicted_population_absent_from_input(predictions):
    only_outfield = predictions[predictions["population"] == "outfield"]
    with pytest.raises(VisualizationDataError, match="goalkeeper"):
        plot_actual_vs_predicted(only_outfield, "goalkeeper")


def test_actual_vs_predicted_rejects_non_finite(predictions):
    broken = predictions.copy()
    broken.loc[0, "actual_overall"] = np.inf
    with pytest.raises(VisualizationDataError, match="NaN or infinite"):
        plot_actual_vs_predicted(broken)


def test_plots_reject_non_dataframe():
    with pytest.raises(VisualizationDataError, match="DataFrame"):
        plot_error_distribution([1, 2, 3])  # type: ignore[arg-type]


# ----------------------------------------------------- error distribution


def test_error_distribution_both_populations(predictions):
    figure = plot_error_distribution(predictions)

    assert_figure(figure)
    ax = figure.axes[0]
    assert ax.get_xlabel() == "Signed error (predicted - actual)"
    assert ax.get_ylabel() == "Density"
    assert "Error distribution" in ax.get_title()
    legend_text = legend_texts(ax)
    assert "zero error" in legend_text
    assert any(text.startswith("goalkeeper mean error") for text in legend_text)
    assert any(text.startswith("outfield mean error") for text in legend_text)
    vertical_x = sorted(float(xdata(line)[0]) for line in ax.lines)
    gk_mean = np.asarray(predictions.loc[predictions["population"] == "goalkeeper", "error"]).mean()
    assert any(np.isclose(value, gk_mean) for value in vertical_x)
    assert any(np.isclose(value, 0.0) for value in vertical_x)


def test_error_distribution_single_population_uses_counts(predictions):
    figure = plot_error_distribution(predictions, "outfield")

    ax = figure.axes[0]
    assert ax.get_ylabel() == "Test rows"
    assert sum(bar_heights(ax)) == OF_ROWS
    assert "outfield" in ax.get_title()


def test_error_distribution_invalid_population(predictions):
    with pytest.raises(VisualizationDataError, match="Unknown population"):
        plot_error_distribution(predictions, "striker")


def test_error_distribution_missing_columns(predictions):
    with pytest.raises(VisualizationDataError, match="error"):
        plot_error_distribution(predictions.drop(columns=["error"]))


def test_error_distribution_empty_input(predictions):
    with pytest.raises(VisualizationDataError, match="No prediction rows"):
        plot_error_distribution(predictions.iloc[0:0])


# ------------------------------------------------------ MAE by rating range


def test_mae_by_rating_range(evaluation_data):
    ranges = get_rating_ranges(evaluation_data, "outfield")
    figure = plot_mae_by_rating_range(ranges, "outfield")

    assert_figure(figure)
    ax = figure.axes[0]
    assert ax.get_xlabel() == "Actual overall rating range"
    assert ax.get_ylabel() == "MAE (rating points)"
    assert ax.get_title() == "MAE by rating range: outfield"
    assert bar_heights(ax) == pytest.approx(list(ranges["mae"]))
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [str(v) for v in ranges["rating_range"]]
    annotations = [text.get_text() for text in ax.texts]
    for count in ranges["samples"]:
        assert any(f"n={int(count):,}" in note for note in annotations)


def test_mae_by_rating_range_flags_small_groups():
    ranges = [
        {"rating_range": "60-69", "samples": 500, "mae": 0.7},
        {"rating_range": "90+", "samples": 3, "mae": 2.2},
    ]
    figure = plot_mae_by_rating_range(ranges, "outfield")

    ax = figure.axes[0]
    big, small = ax.patches
    assert big.get_hatch() is None
    assert small.get_hatch() == "//"
    annotations = [text.get_text() for text in ax.texts]
    assert "small sample" not in annotations[0]
    assert "small sample" in annotations[1]
    assert ax.get_legend() is not None


def test_mae_by_rating_range_no_legend_without_small_groups():
    ranges = [{"rating_range": "60-69", "samples": 500, "mae": 0.7}]
    figure = plot_mae_by_rating_range(ranges, "goalkeeper")
    assert figure.axes[0].get_legend() is None


def test_mae_by_rating_range_does_not_mutate_input():
    ranges = pd.DataFrame(
        {"rating_range": ["60-69", "90+"], "samples": [500, 3], "mae": [0.7, 2.2]}
    )
    before = ranges.copy(deep=True)
    plot_mae_by_rating_range(ranges, "outfield")
    pd.testing.assert_frame_equal(ranges, before)


def test_mae_by_rating_range_invalid_population():
    ranges = [{"rating_range": "60-69", "samples": 5, "mae": 0.7}]
    with pytest.raises(VisualizationDataError, match="Unknown population"):
        plot_mae_by_rating_range(ranges, "striker")


def test_mae_by_rating_range_missing_columns():
    with pytest.raises(VisualizationDataError, match="mae"):
        plot_mae_by_rating_range([{"rating_range": "60-69", "samples": 5}], "outfield")


def test_mae_by_rating_range_empty_input():
    with pytest.raises(VisualizationDataError, match="empty"):
        plot_mae_by_rating_range([], "outfield")
    with pytest.raises(VisualizationDataError, match="empty"):
        plot_mae_by_rating_range(
            pd.DataFrame({"rating_range": [], "samples": [], "mae": []}), "outfield"
        )


@pytest.mark.parametrize(
    "record",
    [
        {"rating_range": "x", "samples": 5, "mae": np.nan},
        {"rating_range": "x", "samples": 5, "mae": -1.0},
        {"rating_range": "x", "samples": 0, "mae": 1.0},
    ],
)
def test_mae_by_rating_range_invalid_values(record):
    with pytest.raises(VisualizationDataError):
        plot_mae_by_rating_range([record], "outfield")


def test_mae_by_rating_range_rejects_wrong_type():
    with pytest.raises(VisualizationDataError, match="DataFrame or a list"):
        plot_mae_by_rating_range("not records", "outfield")  # type: ignore[arg-type]


# ----------------------------------------------------------- training history


def test_training_history_plot():
    history = make_history(epochs=5, best_epoch=4)
    figure = plot_training_history(history, "goalkeeper")

    assert_figure(figure, axes_count=2)
    loss_ax, mae_ax = figure.axes
    assert loss_ax.get_xlabel() == mae_ax.get_xlabel() == "Epoch"
    assert "loss" in loss_ax.get_title().lower()
    assert "MAE" in mae_ax.get_title()
    series = {line.get_label(): line for line in loss_ax.lines + mae_ax.lines}
    assert list(xdata(series["training loss"])) == [1, 2, 3, 4, 5]
    assert list(ydata(series["training loss"])) == pytest.approx(history["train_loss"])
    assert list(ydata(series["validation loss"])) == pytest.approx(history["validation_loss"])
    assert list(ydata(series["validation MAE"])) == pytest.approx(history["validation_mae"])
    best_lines = [line for line in loss_ax.lines + mae_ax.lines if "best epoch" in str(line.get_label())]
    assert len(best_lines) == 2
    assert all(float(xdata(line)[0]) == 4.0 for line in best_lines)
    assert "goalkeeper" in figure._suptitle.get_text()  # type: ignore[union-attr]
    assert "5 epochs" in figure._suptitle.get_text()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("stopped", "expected"), [(True, "yes"), (False, "no"), (None, "not recorded")]
)
def test_training_history_reports_stored_early_stopping(stopped, expected):
    history = make_history()
    history["stopped_early"] = stopped
    figure = plot_training_history(history, "outfield")
    assert f"early stopping recorded: {expected}" in figure._suptitle.get_text()  # type: ignore[union-attr]


def test_training_history_does_not_mutate_input():
    history = make_history()
    before = json.loads(json.dumps(history))
    plot_training_history(history, "outfield")
    assert history == before


def test_training_history_invalid_population():
    with pytest.raises(VisualizationDataError, match="Unknown population"):
        plot_training_history(make_history(), "striker")


@pytest.mark.parametrize("missing", ["train_loss", "validation_loss", "validation_mae", "best_epoch"])
def test_training_history_missing_keys(missing):
    history = make_history()
    del history[missing]
    with pytest.raises(VisualizationDataError):
        plot_training_history(history, "outfield")


def test_training_history_empty():
    history = make_history()
    for key in ("train_loss", "validation_loss", "validation_mae"):
        history[key] = []
    with pytest.raises(VisualizationDataError, match="no epochs"):
        plot_training_history(history, "outfield")


def test_training_history_unequal_lengths():
    history = make_history()
    history["validation_mae"] = history["validation_mae"][:-1]
    with pytest.raises(VisualizationDataError, match="equal lengths"):
        plot_training_history(history, "outfield")


@pytest.mark.parametrize("best_epoch", [0, 6, -1, 2.5, None, True])
def test_training_history_invalid_best_epoch(best_epoch):
    history = make_history(epochs=5)
    history["best_epoch"] = best_epoch
    with pytest.raises(VisualizationDataError, match="best_epoch"):
        plot_training_history(history, "outfield")


def test_training_history_non_finite():
    history = make_history()
    history["train_loss"][0] = float("nan")
    with pytest.raises(VisualizationDataError, match="NaN or infinite"):
        plot_training_history(history, "outfield")


def test_training_history_rejects_non_mapping():
    with pytest.raises(VisualizationDataError, match="mapping"):
        plot_training_history([1, 2], "outfield")  # type: ignore[arg-type]


def test_training_history_non_positive_loss_uses_linear_scale():
    history = make_history()
    history["train_loss"][0] = 0.0
    figure = plot_training_history(history, "outfield")
    assert figure.axes[0].get_yscale() == "linear"
    assert make_history() and plot_training_history(make_history(), "outfield").axes[0].get_yscale() == "log"


# ------------------------------------------------------ population comparison


def test_population_comparison(evaluation_data):
    comparison = get_comparison(evaluation_data)
    figure = plot_population_comparison(comparison)

    assert_figure(figure, axes_count=3)
    titles = [ax.get_title() for ax in figure.axes]
    assert titles[0].startswith("MAE")
    assert titles[1].startswith("RMSE")
    assert titles[2].startswith("R\u00b2")
    for ax, key in zip(figure.axes, ("mae", "rmse", "r2")):
        heights = bar_heights(ax)
        expected = [comparison["goalkeeper"][key], comparison["outfield"][key]]
        assert heights == pytest.approx(expected)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == [
            f"goalkeeper\nn={comparison['goalkeeper']['test_samples']:,}",
            f"outfield\nn={comparison['outfield']['test_samples']:,}",
        ]
    assert "Descriptive" in figure._suptitle.get_text()  # type: ignore[union-attr]


def test_population_comparison_does_not_mutate_input(evaluation_data):
    comparison = get_comparison(evaluation_data)
    before = json.loads(json.dumps(comparison))
    plot_population_comparison(comparison)
    assert comparison == before


def test_population_comparison_missing_population(evaluation_data):
    comparison = get_comparison(evaluation_data)
    del comparison["outfield"]
    with pytest.raises(VisualizationDataError, match="outfield"):
        plot_population_comparison(comparison)


@pytest.mark.parametrize("missing", ["mae", "rmse", "r2", "test_samples"])
def test_population_comparison_missing_metric(evaluation_data, missing):
    comparison = get_comparison(evaluation_data)
    del comparison["goalkeeper"][missing]
    with pytest.raises(VisualizationDataError, match=missing):
        plot_population_comparison(comparison)


def test_population_comparison_empty_and_wrong_type():
    with pytest.raises(VisualizationDataError):
        plot_population_comparison({})
    with pytest.raises(VisualizationDataError, match="mapping"):
        plot_population_comparison([])  # type: ignore[arg-type]


def test_population_comparison_non_finite(evaluation_data):
    comparison = get_comparison(evaluation_data)
    comparison["goalkeeper"]["mae"] = float("inf")
    with pytest.raises(VisualizationDataError, match="NaN or infinite"):
        plot_population_comparison(comparison)


# ------------------------------------------------------------- largest errors


def test_largest_errors(evaluation_data):
    records = pd.concat(
        [
            get_largest_errors(evaluation_data, "goalkeeper"),
            get_largest_errors(evaluation_data, "outfield"),
        ],
        ignore_index=True,
    )
    figure = plot_largest_errors(records, limit=4)

    assert_figure(figure)
    ax = figure.axes[0]
    expected = records.sort_values("absolute_error", ascending=False).head(4).reset_index(drop=True)
    assert bar_widths(ax) == pytest.approx(list(expected["error"]))
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == [
        f"{row['population']} \u00b7 id {int(row['player_id'])}"
        for row in expected.to_dict(orient="records")
    ]
    assert ax.get_xlabel() == "Signed error (predicted - actual)"
    assert "Largest 4" in ax.get_title()
    annotations = [text.get_text() for text in ax.texts]
    assert len(annotations) == 4
    first = expected.iloc[0]
    assert f"actual {first['actual_overall']:.0f}" in annotations[0]
    assert f"predicted {first['predicted_overall']:.2f}" in annotations[0]
    assert f"error {first['error']:+.2f}" in annotations[0]
    assert ax.yaxis_inverted()
    assert ax.get_legend() is not None


def test_largest_errors_limit_larger_than_data(evaluation_data):
    records = get_largest_errors(evaluation_data, "outfield")
    figure = plot_largest_errors(records, limit=50)
    assert len(figure.axes[0].patches) == len(records)


def test_largest_errors_accepts_record_list(evaluation_data):
    records = get_largest_errors(evaluation_data, "goalkeeper").to_dict(orient="records")
    figure = plot_largest_errors(records, limit=2)
    assert len(figure.axes[0].patches) == 2


def test_largest_errors_does_not_mutate_input(evaluation_data):
    records = pd.concat(
        [
            get_largest_errors(evaluation_data, "outfield"),
            get_largest_errors(evaluation_data, "goalkeeper"),
        ],
        ignore_index=True,
    )
    before = records.copy(deep=True)
    plot_largest_errors(records, limit=3)
    pd.testing.assert_frame_equal(records, before)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_largest_errors_invalid_limit(evaluation_data, limit):
    records = get_largest_errors(evaluation_data, "outfield")
    with pytest.raises(VisualizationDataError, match="limit"):
        plot_largest_errors(records, limit=limit)


def test_largest_errors_missing_columns(evaluation_data):
    records = get_largest_errors(evaluation_data, "outfield").drop(columns=["player_id"])
    with pytest.raises(VisualizationDataError, match="player_id"):
        plot_largest_errors(records)


def test_largest_errors_empty_input():
    with pytest.raises(VisualizationDataError, match="empty"):
        plot_largest_errors([])
    with pytest.raises(VisualizationDataError, match="empty"):
        plot_largest_errors(pd.DataFrame(columns=list(PREDICTION_COLUMNS)))


def test_largest_errors_invalid_population_value(evaluation_data):
    records = get_largest_errors(evaluation_data, "outfield")
    records.loc[0, "population"] = "striker"
    with pytest.raises(VisualizationDataError, match="striker"):
        plot_largest_errors(records)


# ------------------------------------------------------- cross-cutting checks


def test_plots_do_not_mutate_prediction_frame(predictions):
    before = predictions.copy(deep=True)

    plot_actual_vs_predicted(predictions)
    plot_actual_vs_predicted(predictions, "goalkeeper", max_points=5)
    plot_error_distribution(predictions)
    plot_error_distribution(predictions, "outfield")
    plot_largest_errors(predictions, limit=3)

    pd.testing.assert_frame_equal(predictions, before)


def test_figures_close_cleanly_and_leave_no_pyplot_state(predictions, evaluation_data):
    assert plt.get_fignums() == []
    figures = [
        plot_actual_vs_predicted(predictions),
        plot_error_distribution(predictions),
        plot_mae_by_rating_range(get_rating_ranges(evaluation_data, "outfield"), "outfield"),
        plot_training_history(get_training_history(evaluation_data, "outfield"), "outfield"),
        plot_population_comparison(get_comparison(evaluation_data)),
        plot_largest_errors(predictions, limit=3),
    ]

    assert plt.get_fignums() == []
    for figure in figures:
        assert isinstance(figure, Figure)
        plt.close(figure)  # must be harmless for figures pyplot never saw
        figure.clear()
        assert figure.axes == []
    assert plt.get_fignums() == []


def test_style_does_not_change_global_rcparams(predictions):
    before = dict(matplotlib.rcParams)
    plot_error_distribution(predictions)
    assert dict(matplotlib.rcParams) == before


def test_public_api_is_explicit():
    import src.visualization as visualization

    assert sorted(visualization.__all__) == sorted(
        [
            "EvaluationData",
            "VisualizationDataError",
            "get_comparison",
            "get_largest_errors",
            "get_population_predictions",
            "get_rating_ranges",
            "get_training_history",
            "load_evaluation_data",
            "plot_actual_vs_predicted",
            "plot_error_distribution",
            "plot_largest_errors",
            "plot_mae_by_rating_range",
            "plot_population_comparison",
            "plot_training_history",
        ]
    )


# ------------------------------------------------------------ architecture


FORBIDDEN_IMPORTS = (
    "torch",
    "src.training",
    "src.experiments",
    "src.data.data_loader",
    "src.evaluation.v1_evaluation",
    "seaborn",
)
PLOT_MODULES = ("performance.py", "training.py", "comparison.py", "style.py")
IO_CALLS = {"open", "read_csv", "read_json", "read_text", "write_text", "to_csv", "to_json"}
FORBIDDEN_CALLS = IO_CALLS | {"savefig", "show"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(VISUALIZATION_DIR.glob("*.py")), ids=lambda p: p.name)
def test_visualization_sources_avoid_forbidden_imports(path):
    for module in _imported_modules(path):
        for forbidden in FORBIDDEN_IMPORTS:
            assert module != forbidden and not module.startswith(forbidden + "."), (
                f"{path.name} imports {module}"
            )
        assert module != "matplotlib.pyplot", f"{path.name} must not use pyplot"


@pytest.mark.parametrize("name", PLOT_MODULES)
def test_plot_modules_do_no_io_or_display(name):
    tree = ast.parse((VISUALIZATION_DIR / name).read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    assert called.isdisjoint(FORBIDDEN_CALLS), called & FORBIDDEN_CALLS


def test_importing_visualization_does_not_import_torch_or_pyplot():
    code = (
        "import sys\n"
        "import src.visualization\n"
        "bad = [m for m in ('torch', 'seaborn', 'matplotlib.pyplot', 'src.training', "
        "'src.experiments', 'src.data.data_loader', 'src.evaluation.v1_evaluation') "
        "if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == ""


# --------------------------------------------- committed artifacts (optional)


COMMITTED_RESULTS = data_module.DEFAULT_RESULTS_PATH
COMMITTED_PREDICTIONS = data_module.DEFAULT_PREDICTIONS_PATH


@pytest.mark.skipif(
    not (COMMITTED_RESULTS.exists() and COMMITTED_PREDICTIONS.exists()),
    reason="committed evaluation artifacts are not present",
)
def test_committed_artifacts_build_all_figures():
    data = load_evaluation_data()
    figures = [
        plot_actual_vs_predicted(data.predictions, max_points=2000),
        plot_error_distribution(data.predictions),
        plot_mae_by_rating_range(get_rating_ranges(data, "outfield"), "outfield"),
        plot_training_history(get_training_history(data, "goalkeeper"), "goalkeeper"),
        plot_population_comparison(get_comparison(data)),
        plot_largest_errors(get_largest_errors(data, "outfield")),
    ]
    assert all(isinstance(figure, Figure) for figure in figures)
