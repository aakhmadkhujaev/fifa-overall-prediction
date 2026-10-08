"""Reusable visualization layer built on the committed V1 evaluation artifacts.

Plot functions take prepared evaluation data and return matplotlib ``Figure``
objects; only ``load_evaluation_data`` touches the filesystem.
"""

from src.visualization.comparison import plot_largest_errors, plot_population_comparison
from src.visualization.data import (
    EvaluationData,
    VisualizationDataError,
    filter_predictions,
    get_comparison,
    get_largest_errors,
    get_metrics,
    get_population_predictions,
    get_prediction_ranges,
    get_rating_ranges,
    get_run_summary,
    get_training_history,
    get_training_summary,
    load_evaluation_data,
)
from src.visualization.performance import (
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_mae_by_rating_range,
)
from src.visualization.training import plot_training_history

__all__ = [
    "EvaluationData",
    "VisualizationDataError",
    "filter_predictions",
    "get_comparison",
    "get_largest_errors",
    "get_metrics",
    "get_population_predictions",
    "get_prediction_ranges",
    "get_rating_ranges",
    "get_run_summary",
    "get_training_history",
    "get_training_summary",
    "load_evaluation_data",
    "plot_actual_vs_predicted",
    "plot_error_distribution",
    "plot_largest_errors",
    "plot_mae_by_rating_range",
    "plot_population_comparison",
    "plot_training_history",
]
