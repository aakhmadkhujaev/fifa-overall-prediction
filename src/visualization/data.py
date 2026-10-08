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
