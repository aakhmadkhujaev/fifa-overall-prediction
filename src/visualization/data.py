"""Load and validate the committed evaluation artifacts used for visualization.

This is the only visualization module that performs filesystem I/O. It reads
``reports/generated/v1_final_evaluation.json`` and
``reports/generated/v1_test_predictions.csv`` relative to the project root and
never follows the absolute dataset or checkpoint paths stored inside the JSON.

The remaining helpers are pure: they validate in-memory data so every plot
function can fail with a clear :class:`VisualizationDataError`.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import PREDICTION_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "generated" / "v1_final_evaluation.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "generated" / "v1_test_predictions.csv"
EXPECTED_SCHEMA_VERSION = 1
POPULATIONS = ("goalkeeper", "outfield")
NUMERIC_PREDICTION_COLUMNS = tuple(column for column in PREDICTION_COLUMNS if column != "population")
METRIC_KEYS = (
    "mae",
    "rmse",
    "r2",
    "mean_error",
    "underprediction_count",
    "overprediction_count",
    "zero_error_count",
    "underprediction_rate",
    "overprediction_rate",
    "zero_error_rate",
    "test_samples",
)
DIRECTIONS = ("all", "under", "over")
PathLike = Union[str, Path]


class VisualizationDataError(ValueError):
    """Raised when evaluation data is missing, malformed, or inconsistent."""


@dataclass(frozen=True, eq=False)
class EvaluationData:
    """Validated evaluation artifacts. Use the ``get_*`` accessors to read it."""

    predictions: pd.DataFrame
    results: Mapping[str, Any]


def check_population(population: object, *, allow_none: bool = False) -> None:
    """Raise unless ``population`` is a known population (or ``None`` if allowed)."""
    if population is None and allow_none:
        return
    if not isinstance(population, str) or population not in POPULATIONS:
        raise VisualizationDataError(
            f"Unknown population {population!r}; expected one of {list(POPULATIONS)}."
        )


def as_finite_array(values: object, name: str) -> np.ndarray:
    """Return ``values`` as a 1-D finite float array or raise."""
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise VisualizationDataError(f"{name} must be numeric.") from exc
    if not np.isfinite(array).all():
        raise VisualizationDataError(f"{name} contains NaN or infinite values.")
    return array


def validate_prediction_frame(
    predictions: object,
    required: Iterable[str] = PREDICTION_COLUMNS,
) -> None:
    """Check required columns, numeric finiteness, and population values."""
    if not isinstance(predictions, pd.DataFrame):
        raise VisualizationDataError("Predictions must be a pandas DataFrame.")
    required_columns = list(required)
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise VisualizationDataError(f"Prediction frame is missing columns: {missing}")
    for column in required_columns:
        if column == "population":
            continue
        series = predictions[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise VisualizationDataError(f"Column {column!r} must be numeric.")
        if not np.isfinite(series.to_numpy(dtype=np.float64)).all():
            raise VisualizationDataError(f"Column {column!r} contains NaN or infinite values.")
    if "population" in required_columns:
        invalid = sorted(
            {str(value) for value in predictions["population"].unique() if value not in POPULATIONS}
        )
        if invalid:
            raise VisualizationDataError(
                f"Unknown population values {invalid}; expected {list(POPULATIONS)}."
            )


def coerce_records_frame(
    records: object,
    required: Iterable[str],
    name: str,
) -> pd.DataFrame:
    """Return a new DataFrame from a DataFrame or a sequence of record mappings."""
    required_columns = list(required)
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, (list, tuple)):
        try:
            frame = pd.DataFrame(list(records))
        except (TypeError, ValueError) as exc:
            raise VisualizationDataError(f"{name} records are malformed.") from exc
    else:
        raise VisualizationDataError(f"{name} must be a DataFrame or a list of records.")
    if frame.empty:
        raise VisualizationDataError(f"{name} is empty.")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise VisualizationDataError(f"{name} is missing columns: {missing}")
    return frame


def _require_mapping(container: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise VisualizationDataError(f"{context} is missing the mapping {key!r}.")
    return value


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise VisualizationDataError(f"Evaluation JSON not found: {path}") from None
    except OSError as exc:
        raise VisualizationDataError(f"Evaluation JSON could not be read: {path}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise VisualizationDataError(f"Evaluation JSON is not valid JSON: {path}") from exc


def _read_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise VisualizationDataError(f"Prediction CSV not found: {path}") from None
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise VisualizationDataError(f"Prediction CSV could not be read: {path}") from exc


def _validate_results(results: object) -> Mapping[str, Any]:
    if not isinstance(results, Mapping):
        raise VisualizationDataError("Evaluation JSON must contain an object at the top level.")
    version = results.get("schema_version")
    if isinstance(version, bool) or version != EXPECTED_SCHEMA_VERSION:
        raise VisualizationDataError(
            f"Unsupported schema_version {version!r}; expected {EXPECTED_SCHEMA_VERSION}."
        )
    populations = _require_mapping(results, "populations", "Evaluation JSON")
    comparison = _require_mapping(results, "comparison", "Evaluation JSON")
    for population in POPULATIONS:
        context = f"populations[{population!r}]"
        entry = _require_mapping(populations, population, "Evaluation JSON")
        metrics = _require_mapping(entry, "metrics", context)
        error_analysis = _require_mapping(entry, "error_analysis", context)
        _require_mapping(entry, "training_history", context)
        _require_mapping(comparison, population, "Evaluation JSON comparison")
        test_samples = metrics.get("test_samples")
        if isinstance(test_samples, bool) or not isinstance(test_samples, int):
            raise VisualizationDataError(f"{context} metrics need an integer 'test_samples'.")
        for key in ("rating_ranges", "largest_absolute_errors"):
            if not isinstance(error_analysis.get(key), list):
                raise VisualizationDataError(f"{context} error_analysis is missing list {key!r}.")
    return results


def _validate_loaded_predictions(predictions: pd.DataFrame, results: Mapping[str, Any]) -> None:
    if list(predictions.columns) != list(PREDICTION_COLUMNS):
        raise VisualizationDataError(
            f"Prediction CSV columns {list(predictions.columns)} do not match "
            f"the expected {list(PREDICTION_COLUMNS)}."
        )
    if predictions.empty:
        raise VisualizationDataError("Prediction CSV contains no rows.")
    validate_prediction_frame(predictions)
    for population in POPULATIONS:
        actual_rows = int((predictions["population"] == population).sum())
        if actual_rows == 0:
            raise VisualizationDataError(f"Prediction CSV has no rows for {population!r}.")
        expected_rows = results["populations"][population]["metrics"]["test_samples"]
        if actual_rows != expected_rows:
            raise VisualizationDataError(
                f"Prediction CSV has {actual_rows} {population!r} rows but the evaluation "
                f"JSON reports {expected_rows} test samples."
            )


def load_evaluation_data(
    results_path: PathLike | None = None,
    predictions_path: PathLike | None = None,
) -> EvaluationData:
    """Load and validate the committed evaluation JSON and prediction CSV."""
    results = _validate_results(
        _read_json(Path(results_path) if results_path is not None else DEFAULT_RESULTS_PATH)
    )
    predictions = _read_predictions(
        Path(predictions_path) if predictions_path is not None else DEFAULT_PREDICTIONS_PATH
    )
    _validate_loaded_predictions(predictions, results)
    return EvaluationData(predictions=predictions, results=results)


def get_population_predictions(data: EvaluationData, population: str) -> pd.DataFrame:
    """Return a copy of one population's held-out prediction rows."""
    check_population(population)
    mask = data.predictions["population"] == population
    return pd.DataFrame(data.predictions.loc[mask]).reset_index(drop=True)


def get_rating_ranges(data: EvaluationData, population: str) -> pd.DataFrame:
    """Return the stored MAE-by-rating-range summary for one population."""
    check_population(population)
    records = data.results["populations"][population]["error_analysis"]["rating_ranges"]
    return pd.DataFrame(copy.deepcopy(records))


def get_training_history(data: EvaluationData, population: str) -> dict[str, Any]:
    """Return a copy of the stored training history for one population."""
    check_population(population)
    return copy.deepcopy(dict(data.results["populations"][population]["training_history"]))


def get_comparison(data: EvaluationData) -> dict[str, Any]:
    """Return a copy of the stored goalkeeper-vs-outfield metric comparison."""
    return copy.deepcopy(dict(data.results["comparison"]))


def get_largest_errors(data: EvaluationData, population: str) -> pd.DataFrame:
    """Return the stored largest absolute errors for one population."""
    check_population(population)
    records = data.results["populations"][population]["error_analysis"]["largest_absolute_errors"]
    return pd.DataFrame(copy.deepcopy(records))


def get_metrics(data: EvaluationData, population: str) -> dict[str, float | int]:
    """Return the stored test metrics and error-direction summary for one population."""
    check_population(population)
    stored = data.results["populations"][population]["metrics"]
    missing = [key for key in METRIC_KEYS if key not in stored]
    if missing:
        raise VisualizationDataError(f"Stored metrics for {population!r} are missing: {missing}")
    for key in METRIC_KEYS:
        as_finite_array(stored[key], f"metrics[{key!r}]")
    return {key: copy.deepcopy(stored[key]) for key in METRIC_KEYS}


def get_run_summary(data: EvaluationData) -> dict[str, Any]:
    """Return portable run facts: dataset size, split settings, per-population sizes.

    Machine-specific paths stored in the JSON are never included.
    """
    evaluation = data.results.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise VisualizationDataError("Evaluation JSON is missing the 'evaluation' section.")
    summary: dict[str, Any] = {"schema_version": data.results["schema_version"]}
    for key in ("dataset_rows", "test_size", "validation_size", "split_random_state"):
        if key not in evaluation:
            raise VisualizationDataError(f"Evaluation section is missing {key!r}.")
        summary[key] = copy.deepcopy(evaluation[key])
    populations: dict[str, dict[str, Any]] = {}
    for population in POPULATIONS:
        entry = data.results["populations"][population]
        split_sizes = entry.get("split_sizes")
        metadata = entry.get("checkpoint_metadata")
        if not isinstance(split_sizes, Mapping) or not isinstance(metadata, Mapping):
            raise VisualizationDataError(
                f"populations[{population!r}] needs split_sizes and checkpoint_metadata."
            )
        try:
            populations[population] = {
                "train_size": split_sizes["train_size"],
                "validation_size": split_sizes["validation_size"],
                "test_size": split_sizes["test_size"],
                "feature_count": metadata["input_feature_count"],
            }
        except KeyError as exc:
            raise VisualizationDataError(
                f"populations[{population!r}] is missing {exc.args[0]!r}."
            ) from None
    summary["populations"] = populations
    return summary


def get_training_summary(data: EvaluationData, population: str) -> dict[str, Any]:
    """Return the stored epoch count, best epoch, best validation MAE and early-stop flag.

    Values are reported exactly as stored; nothing is inferred from epoch counts.
    """
    history = get_training_history(data, population)
    for key in ("train_loss", "best_epoch", "best_validation_mae"):
        if key not in history:
            raise VisualizationDataError(f"Stored training history is missing {key!r}.")
    return {
        "epochs_stored": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "best_validation_mae": history["best_validation_mae"],
        "stopped_early": history.get("stopped_early"),
    }


def get_prediction_ranges(
    predictions: pd.DataFrame,
    population: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Return outer bounds for the filterable columns, suitable for slider limits.

    Rating bounds are rounded outward to whole points and the absolute-error upper
    bound is rounded up to two decimals.
    """
    rows = _filter_population(predictions, population)
    if rows.empty:
        raise VisualizationDataError("No prediction rows available for the requested population.")
    actual = rows["actual_overall"].to_numpy(dtype=np.float64)
    absolute_error = rows["absolute_error"].to_numpy(dtype=np.float64)
    return {
        "actual_overall": (float(math.floor(actual.min())), float(math.ceil(actual.max()))),
        "absolute_error": (0.0, math.ceil(float(absolute_error.max()) * 100) / 100),
    }


def _filter_population(predictions: pd.DataFrame, population: str | None) -> pd.DataFrame:
    check_population(population, allow_none=True)
    validate_prediction_frame(predictions, PREDICTION_COLUMNS)
    if population is None:
        return predictions
    return pd.DataFrame(predictions.loc[predictions["population"] == population])


def _validated_range(bounds: object, name: str) -> tuple[float, float]:
    values = as_finite_array(bounds, name)
    if values.size != 2:
        raise VisualizationDataError(f"{name} must be a (low, high) pair.")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise VisualizationDataError(f"{name} low bound {low} exceeds high bound {high}.")
    return low, high


def filter_predictions(
    predictions: pd.DataFrame,
    population: str | None = None,
    *,
    rating_range: tuple[float, float] | None = None,
    error_range: tuple[float, float] | None = None,
    direction: str = "all",
    player_id: int | str | None = None,
) -> pd.DataFrame:
    """Return a new frame of matching prediction rows, largest absolute error first.

    ``rating_range`` filters ``actual_overall`` and ``error_range`` filters
    ``absolute_error``; both are inclusive. ``direction`` is ``"all"``, ``"under"``
    (error < 0) or ``"over"`` (error > 0). ``player_id`` may be an int or a digit
    string; an empty string means no filter. The input is never modified.
    """
    if direction not in DIRECTIONS:
        raise VisualizationDataError(
            f"Unknown direction {direction!r}; expected one of {list(DIRECTIONS)}."
        )
    rows = _filter_population(predictions, population)
    mask = pd.Series(True, index=rows.index)
    if rating_range is not None:
        low, high = _validated_range(rating_range, "rating_range")
        mask &= rows["actual_overall"].between(low, high)
    if error_range is not None:
        low, high = _validated_range(error_range, "error_range")
        mask &= rows["absolute_error"].between(low, high)
    if direction == "under":
        mask &= rows["error"] < 0
    elif direction == "over":
        mask &= rows["error"] > 0
    parsed_id = _parse_player_id(player_id)
    if parsed_id is not None:
        mask &= rows["player_id"] == parsed_id
    selected = pd.DataFrame(rows.loc[mask])
    return selected.sort_values("absolute_error", ascending=False, kind="stable").reset_index(
        drop=True
    )


def _parse_player_id(player_id: int | str | None) -> int | None:
    if player_id is None:
        return None
    if isinstance(player_id, bool):
        raise VisualizationDataError("player_id must be an integer.")
    if isinstance(player_id, str):
        text = player_id.strip()
        if not text:
            return None
        if not text.lstrip("+-").isdigit():
            raise VisualizationDataError(f"player_id {player_id!r} is not an integer.")
        return int(text)
    try:
        if int(player_id) != player_id:
            raise ValueError
    except (TypeError, ValueError):
        raise VisualizationDataError(f"player_id {player_id!r} is not an integer.") from None
    return int(player_id)
