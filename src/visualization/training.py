"""Training-history plot built from the history stored in the evaluation JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.visualization.data import VisualizationDataError, as_finite_array, check_population
from src.visualization.style import NEUTRAL_COLOR, new_figure, plot_style, population_color

_SERIES_KEYS = ("train_loss", "validation_loss", "validation_mae")


def plot_training_history(
    training_history: Mapping[str, Any],
    population_label: str,
) -> Figure:
    """Plot stored training/validation loss and validation MAE per epoch.

    Only the stored history is drawn. The best epoch is the one recorded in the
    checkpoint; the figure makes no claim about convergence or overfitting.
    """
    check_population(population_label)
    if not isinstance(training_history, Mapping):
        raise VisualizationDataError("training_history must be a mapping.")
    series: dict[str, np.ndarray] = {}
    for key in _SERIES_KEYS:
        if key not in training_history:
            raise VisualizationDataError(f"training_history is missing {key!r}.")
        series[key] = as_finite_array(training_history[key], f"training_history[{key!r}]")
    epochs_stored = len(series["train_loss"])
    if epochs_stored == 0:
        raise VisualizationDataError("training_history contains no epochs.")
    if any(len(values) != epochs_stored for values in series.values()):
        raise VisualizationDataError("training_history series must have equal lengths.")
    best_epoch = training_history.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, (int, np.integer))
        or not 1 <= best_epoch <= epochs_stored
    ):
        raise VisualizationDataError(
            f"best_epoch must be an integer between 1 and {epochs_stored}."
        )
    stopped_early = training_history.get("stopped_early")
    stopped_text = "not recorded" if stopped_early is None else ("yes" if stopped_early else "no")

    epochs = np.arange(1, epochs_stored + 1)
    color = population_color(population_label)
    with plot_style():
        figure, axes = new_figure(1, 2, figsize=(11.0, 4.5))
        loss_ax, mae_ax = axes[0, 0], axes[0, 1]
        loss_ax.plot(epochs, series["train_loss"], color=color, label="training loss")
        loss_ax.plot(
            epochs, series["validation_loss"], color=color, linestyle="--", label="validation loss"
        )
        if min(series["train_loss"].min(), series["validation_loss"].min()) > 0:
            loss_ax.set_yscale("log")
            loss_ax.set_ylabel("Loss (log scale)")
        else:
            loss_ax.set_ylabel("Loss")
        loss_ax.set_title("Training and validation loss")
        mae_ax.plot(epochs, series["validation_mae"], color=color, label="validation MAE")
        mae_ax.set_ylabel("MAE (rating points)")
        mae_ax.set_title("Validation MAE")
        for ax in (loss_ax, mae_ax):
            ax.axvline(
                float(best_epoch),
                color=NEUTRAL_COLOR,
                linestyle=":",
                linewidth=1.5,
                label=f"best epoch (stored): {int(best_epoch)}",
            )
            ax.set_xlabel("Epoch")
            ax.legend()
        figure.suptitle(
            f"Training history: {population_label}\n"
            f"Stored history: {epochs_stored} epochs; early stopping recorded: {stopped_text}"
        )
    return figure
