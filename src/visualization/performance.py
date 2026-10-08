"""Test-set performance plots built from prepared evaluation data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.visualization.data import (
    POPULATIONS,
    VisualizationDataError,
    as_finite_array,
    check_population,
    coerce_records_frame,
    validate_prediction_frame,
)
from src.visualization.style import (
    NEUTRAL_COLOR,
    new_figure,
    plot_style,
    population_color,
)

SUBSAMPLE_SEED = 42
HISTOGRAM_BINS = 40
SMALL_SAMPLE_THRESHOLD = 30


def _select_rows(
    predictions: pd.DataFrame,
    population: str | None,
    required: Sequence[str],
) -> pd.DataFrame:
    """Validate and return a new frame of the requested rows and columns."""
    check_population(population, allow_none=True)
    validate_prediction_frame(predictions, required)
    frame = predictions.loc[:, list(required)]
    if population is not None:
        frame = frame[frame["population"] == population]
    if frame.empty:
        scope = population if population is not None else "any population"
        raise VisualizationDataError(f"No prediction rows available for {scope}.")
    return frame


def _subsample(frame: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None:
        return frame
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points <= 0:
        raise VisualizationDataError("max_points must be a positive integer or None.")
    if len(frame) <= max_points:
        return frame
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    chosen = np.sort(rng.choice(len(frame), size=int(max_points), replace=False))
    return frame.iloc[chosen]


def _scope_label(population: str | None) -> str:
    return population if population is not None else "goalkeeper + outfield"


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    population: str | None = None,
    *,
    max_points: int | None = None,
) -> Figure:
    """Scatter of actual (x) vs predicted (y) overall with a y = x reference line.

    ``max_points`` draws a deterministic random subsample of at most that many rows.
    """
    required = ("actual_overall", "predicted_overall", "population")
    frame = _select_rows(predictions, population, required)
    total_rows = len(frame)
    frame = _subsample(frame, max_points)

    with plot_style():
        figure, axes = new_figure()
        ax = axes[0, 0]
        for name in POPULATIONS:
            rows = frame[frame["population"] == name]
            if rows.empty:
                continue
            ax.scatter(
                rows["actual_overall"],
                rows["predicted_overall"],
                s=6,
                alpha=0.35,
                linewidths=0,
                color=population_color(name),
                label=f"{name} (n={len(rows):,})",
                rasterized=True,
            )
        values = frame[["actual_overall", "predicted_overall"]].to_numpy(dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        ax.plot(
            [low, high],
            [low, high],
            color=NEUTRAL_COLOR,
            linewidth=1,
            linestyle="--",
            label="y = x (perfect prediction)",
        )
        ax.set_xlabel("Actual overall rating")
        ax.set_ylabel("Predicted overall rating")
        shown = f"{len(frame):,} of {total_rows:,} test rows shown"
        ax.set_title(f"Actual vs predicted overall: {_scope_label(population)}\n({shown})")
        ax.set_aspect("equal", adjustable="box")
        legend = ax.legend(loc="upper left")
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
    return figure


def plot_error_distribution(
    predictions: pd.DataFrame,
    population: str | None = None,
) -> Figure:
    """Histogram of signed error (predicted - actual) with zero and mean-error lines.

    When both populations are shown the histograms are density-normalised so the
    much smaller goalkeeper group stays visible.
    """
    required = ("error", "population")
    frame = _select_rows(predictions, population, required)
    overlay = population is None and frame["population"].nunique() > 1
    bins = np.histogram_bin_edges(frame["error"].to_numpy(dtype=np.float64), bins=HISTOGRAM_BINS)

    with plot_style():
        figure, axes = new_figure()
        ax = axes[0, 0]
        for name in POPULATIONS:
            errors = frame.loc[frame["population"] == name, "error"].to_numpy(dtype=np.float64)
            if errors.size == 0:
                continue
            color = population_color(name)
            ax.hist(
                errors,
                bins=bins,
                density=overlay,
                alpha=0.6,
                color=color,
                label=f"{name} (n={errors.size:,})",
            )
            ax.axvline(
                float(errors.mean()),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{name} mean error {errors.mean():+.3f}",
            )
        ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=1, label="zero error")
        ax.set_xlabel("Signed error (predicted - actual)")
        ax.set_ylabel("Density" if overlay else "Test rows")
        ax.set_title(f"Error distribution: {_scope_label(population)}")
        ax.legend()
    return figure


def plot_mae_by_rating_range(
    rating_ranges: pd.DataFrame | Sequence[dict],
    population_label: str,
) -> Figure:
    """Bar chart of the stored MAE per actual-rating band with sample counts.

    Bands with fewer than ``SMALL_SAMPLE_THRESHOLD`` samples are hatched and flagged.
    """
    check_population(population_label)
    frame = coerce_records_frame(
        rating_ranges, ("rating_range", "samples", "mae"), "rating_ranges"
    )
    labels = [str(value) for value in frame["rating_range"]]
    samples = as_finite_array(frame["samples"], "rating_ranges samples")
    mae = as_finite_array(frame["mae"], "rating_ranges mae")
    if (samples < 1).any() or (mae < 0).any():
        raise VisualizationDataError("rating_ranges need samples >= 1 and mae >= 0.")
    small = samples < SMALL_SAMPLE_THRESHOLD
    color = population_color(population_label)

    with plot_style():
        figure, axes = new_figure(figsize=(7.0, 4.5))
        ax = axes[0, 0]
        positions = np.arange(len(labels))
        bars = ax.bar(positions, mae, color=color, edgecolor=NEUTRAL_COLOR, linewidth=0.8)
        for bar, is_small in zip(bars, small):
            if is_small:
                bar.set_alpha(0.45)
                bar.set_hatch("//")
        for position, value, count, is_small in zip(positions, mae, samples, small):
            note = f"MAE {value:.3f}\nn={int(count):,}"
            if is_small:
                note += "\nsmall sample"
            ax.annotate(
                note,
                (position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, float(mae.max()) * 1.35 if mae.max() > 0 else 1.0)
        ax.set_xlabel("Actual overall rating range")
        ax.set_ylabel("MAE (rating points)")
        ax.set_title(f"MAE by rating range: {population_label}")
        if small.any():
            ax.legend(
                handles=[
                    Patch(
                        facecolor=color,
                        alpha=0.45,
                        hatch="//",
                        edgecolor=NEUTRAL_COLOR,
                        label=f"n < {SMALL_SAMPLE_THRESHOLD}: interpret with caution",
                    )
                ],
                loc="upper left",
            )
    return figure
