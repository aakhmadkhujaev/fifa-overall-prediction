"""Goalkeeper-vs-outfield comparison and largest-error plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.evaluation.error_analysis import PREDICTION_COLUMNS
from src.visualization.data import (
    POPULATIONS,
    VisualizationDataError,
    as_finite_array,
    coerce_records_frame,
    validate_prediction_frame,
)
from src.visualization.style import NEUTRAL_COLOR, new_figure, plot_style, population_color

_METRICS = (
    ("mae", "MAE (lower is better)"),
    ("rmse", "RMSE (lower is better)"),
    ("r2", "R² (higher is better)"),
)


def plot_population_comparison(comparison: Mapping[str, Any]) -> Figure:
    """Bar panels of the stored MAE, RMSE and R² for each population, with test sizes."""
    if not isinstance(comparison, Mapping):
        raise VisualizationDataError("comparison must be a mapping.")
    values: dict[str, dict[str, float]] = {}
    for population in POPULATIONS:
        entry = comparison.get(population)
        if not isinstance(entry, Mapping):
            raise VisualizationDataError(f"comparison is missing population {population!r}.")
        row: dict[str, float] = {}
        for key in ("mae", "rmse", "r2", "test_samples"):
            if key not in entry:
                raise VisualizationDataError(f"comparison[{population!r}] is missing {key!r}.")
            row[key] = float(as_finite_array(entry[key], f"comparison[{population!r}][{key!r}]")[0])
        if row["mae"] < 0 or row["rmse"] < 0 or row["test_samples"] < 1:
            raise VisualizationDataError(
                f"comparison[{population!r}] needs mae >= 0, rmse >= 0 and test_samples >= 1."
            )
        values[population] = row

    tick_labels = [f"{name}\nn={int(values[name]['test_samples']):,}" for name in POPULATIONS]
    colors = [population_color(name) for name in POPULATIONS]
    with plot_style():
        figure, axes = new_figure(1, 3, figsize=(11.0, 4.5))
        for index, (key, title) in enumerate(_METRICS):
            ax = axes[0, index]
            heights = [values[name][key] for name in POPULATIONS]
            bars = ax.bar(tick_labels, heights, color=colors, edgecolor=NEUTRAL_COLOR, linewidth=0.8)
            ax.bar_label(bars, fmt="%.4f", padding=3)
            low = min(0.0, min(heights) * 1.1)
            high = max(heights) * 1.2 if key != "r2" else max(1.0, max(heights)) * 1.12
            ax.set_ylim(low, high if high > low else low + 1.0)
            ax.set_title(title)
        figure.suptitle(
            "Goalkeeper vs outfield: held-out test metrics\n"
            "Descriptive comparison of two fixed test distributions"
        )
    return figure


def plot_largest_errors(
    largest_errors: pd.DataFrame | Sequence[dict],
    *,
    limit: int = 10,
) -> Figure:
    """Horizontal bars of the largest absolute errors, shown as signed error.

    Each bar is labelled with population and player_id and annotated with the
    actual rating, predicted rating and signed error.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise VisualizationDataError("limit must be a positive integer.")
    frame = coerce_records_frame(largest_errors, PREDICTION_COLUMNS, "largest_errors")
    validate_prediction_frame(frame, PREDICTION_COLUMNS)
    top = (
        frame.sort_values("absolute_error", ascending=False, kind="stable")
        .head(int(limit))
        .reset_index(drop=True)
    )
    count = len(top)
    positions = np.arange(count)
    errors = top["error"].to_numpy(dtype=np.float64)
    colors = [population_color(name) for name in top["population"]]
    rows = top.to_dict(orient="records")
    y_labels = [f"{row['population']} \u00b7 id {int(row['player_id'])}" for row in rows]

    with plot_style():
        figure, axes = new_figure(figsize=(9.0, 0.5 * count + 2.5))
        ax = axes[0, 0]
        ax.barh(positions, errors, color=colors, edgecolor=NEUTRAL_COLOR, linewidth=0.8)
        ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=1)
        for position, row in zip(positions, rows):
            error = float(row["error"])
            ax.annotate(
                f"actual {row['actual_overall']:.0f} → "
                f"predicted {row['predicted_overall']:.2f} (error {error:+.2f})",
                (error, position),
                xytext=(4 if error >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if error >= 0 else "right",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(positions, y_labels)
        ax.invert_yaxis()
        ax.margins(x=0.45)
        ax.set_xlabel("Signed error (predicted - actual)")
        ax.set_title(f"Largest {count} absolute errors (held-out test set)")
        present = [name for name in POPULATIONS if (top["population"] == name).any()]
        ax.legend(
            handles=[Patch(facecolor=population_color(name), label=name) for name in present],
            loc="lower right",
        )
    return figure
