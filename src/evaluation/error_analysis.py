"""Pure prediction and error-analysis helpers for final model evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = (
    "player_id",
    "actual_overall",
    "predicted_overall",
    "error",
    "absolute_error",
    "population",
)


def build_prediction_frame(
    player_ids: Sequence[object] | pd.Series,
    actual: Sequence[float] | pd.Series | np.ndarray,
    predicted: Sequence[float] | pd.Series | np.ndarray,
    population: str,
) -> pd.DataFrame:
    """Build aligned prediction-level records with signed and absolute errors."""
    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    player_array = np.asarray(player_ids).reshape(-1)
    if not (len(player_array) == len(actual_array) == len(predicted_array)):
        raise ValueError("Player IDs, actual values, and predictions must have equal lengths.")
    if len(actual_array) == 0:
        raise ValueError("At least one prediction is required.")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("Actual values and predictions must be finite.")
    error = predicted_array - actual_array
    return pd.DataFrame(
        {
            "player_id": player_array,
            "actual_overall": actual_array,
            "predicted_overall": predicted_array,
            "error": error,
            "absolute_error": np.abs(error),
            "population": population,
        },
        columns=PREDICTION_COLUMNS,
    )


def calculate_error_direction(
    actual: Sequence[float] | pd.Series | np.ndarray,
    predicted: Sequence[float] | pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate signed-error and under/overprediction summaries."""
    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual_array.size == 0:
        raise ValueError("At least one test observation is required.")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and prediction arrays must have the same shape.")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("Actual values and predictions must be finite.")

    error = predicted_array - actual_array
    return {
        "mean_error": float(error.mean()),
        "underprediction_count": int((error < 0).sum()),
        "overprediction_count": int((error > 0).sum()),
        "zero_error_count": int((error == 0).sum()),
        "underprediction_rate": float((error < 0).mean()),
        "overprediction_rate": float((error > 0).mean()),
        "zero_error_rate": float((error == 0).mean()),
        "test_samples": int(actual_array.size),
    }


def summarize_rating_ranges(
    predictions: pd.DataFrame,
    *,
    bins: Sequence[float] = (-np.inf, 59.999, 69.999, 79.999, 89.999, np.inf),
    labels: Sequence[str] = ("<60", "60-69", "70-79", "80-89", "90+"),
) -> pd.DataFrame:
    """Summarize error by fixed actual-rating bands."""
    required = {"actual_overall", "error"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    frame = predictions.loc[:, ["actual_overall", "error"]].copy()
    frame["rating_range"] = pd.cut(
        frame["actual_overall"], bins=bins, labels=labels, right=True
    )
    summary = frame.groupby("rating_range", observed=False).agg(
        samples=("actual_overall", "size"),
        actual_mean=("actual_overall", "mean"),
        mean_error=("error", "mean"),
        mae=("error", lambda values: np.abs(values).mean()),
        rmse=("error", lambda values: np.sqrt(np.square(values).mean())),
    ).reset_index()
    return summary[summary["samples"] > 0].reset_index(drop=True)


def largest_absolute_errors(
    predictions: pd.DataFrame,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    """Return the largest absolute errors in descending order."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    required = set(PREDICTION_COLUMNS)
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    return (
        predictions.sort_values("absolute_error", ascending=False)
        .loc[:, list(PREDICTION_COLUMNS)]
        .head(limit)
        .reset_index(drop=True)
    )
