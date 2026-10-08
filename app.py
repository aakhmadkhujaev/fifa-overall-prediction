"""Streamlit explorer for the final V1 FIFA Overall Rating evaluation.

The app only lays out controls and results. All data access, filtering and
figures come from ``src.visualization``, which reads the committed evaluation
artifacts. It does not train models, run inference, load checkpoints or read
the raw dataset.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from matplotlib.figure import Figure

from src.visualization import (
    EvaluationData,
    VisualizationDataError,
    filter_predictions,
    get_comparison,
    get_metrics,
    get_prediction_ranges,
    get_rating_ranges,
    get_run_summary,
    get_training_history,
    get_training_summary,
    load_evaluation_data,
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_largest_errors,
    plot_mae_by_rating_range,
    plot_population_comparison,
    plot_training_history,
)
from src.visualization.data import PROJECT_ROOT

SECTIONS = ("Overview", "Performance", "Training", "Error Analysis", "Test Predictions")
POPULATION_CHOICES = {"Both": None, "Goalkeeper": "goalkeeper", "Outfield": "outfield"}
DIRECTION_CHOICES = {
    "All": "all",
    "Underpredicted (error < 0)": "under",
    "Overpredicted (error > 0)": "over",
}
DEFAULT_SCATTER_POINTS = 5000
MIN_SCATTER_POINTS = 500
SCATTER_STEP = 500

PURPOSE = (
    "This app explores the final V1 evaluation of the FIFA player Overall Rating (OVR) "
    "models: one feedforward PyTorch network for goalkeepers and one for outfield players. "
    "It reads the committed evaluation artifacts only. It does not train models, run "
    "inference, or load checkpoints or the raw dataset."
)
SPLIT_CAVEAT = (
    "All results come from one deterministic grouped held-out test split. They describe "
    "these two fixed test sets and do not by themselves explain why the populations differ."
)


@st.cache_resource(show_spinner="Loading evaluation artifacts...")
def _load_data() -> EvaluationData:
    """Load the evaluation artifacts once per server. Treat the result as read-only."""
    return load_evaluation_data()


def _safe_message(error: Exception) -> str:
    """Return an error message without the server's absolute project path."""
    return str(error).replace(str(PROJECT_ROOT), ".")


def _show_figure(build: Callable[[], Figure]) -> None:
    """Build and display one figure, reporting data problems in the page."""
    try:
        figure = build()
    except VisualizationDataError as error:
        st.error(f"This chart could not be drawn: {_safe_message(error)}")
        return
    st.pyplot(figure)


def _selected_populations(population: str | None) -> list[str]:
    return [population] if population is not None else ["goalkeeper", "outfield"]


def _format_flag(value: object) -> str:
    if value is None:
        return "Not recorded"
    return "True" if value else "False"


def _render_sidebar(data: EvaluationData) -> tuple[str, str | None]:
    st.sidebar.title("FIFA OVR V1")
    section = st.sidebar.radio("Section", SECTIONS, key="section")
    label = st.sidebar.radio("Population", list(POPULATION_CHOICES), key="population")
    summary = get_run_summary(data)
    st.sidebar.divider()
    st.sidebar.caption(
        f"Dataset rows: {summary['dataset_rows']:,}  \n"
        f"Test fraction: {summary['test_size']:.0%}, "
        f"validation fraction: {summary['validation_size']:.0%}  \n"
        f"Split random state: {summary['split_random_state']}  \n"
        f"Artifact schema version: {summary['schema_version']}"
    )
    return str(section), POPULATION_CHOICES[str(label)]


def _render_overview(data: EvaluationData, population: str | None) -> None:
    st.header("Overview")
    st.write(PURPOSE)
    st.caption(SPLIT_CAVEAT)
    summary = get_run_summary(data)
    for name in _selected_populations(population):
        metrics = get_metrics(data, name)
        sizes = summary["populations"][name]
        st.subheader(name.title())
        columns = st.columns(5)
        columns[0].metric("Test rows", f"{metrics['test_samples']:,}")
        columns[1].metric(
            "MAE", f"{metrics['mae']:.3f}", help="Mean absolute error, in rating points."
        )
        columns[2].metric("RMSE", f"{metrics['rmse']:.3f}")
        columns[3].metric("R²", f"{metrics['r2']:.4f}")
        columns[4].metric(
            "Mean error",
            f"{metrics['mean_error']:+.3f}",
            help="Mean of predicted minus actual rating.",
        )
        st.caption(
            f"Input features: {sizes['feature_count']} · "
            f"train / validation / test rows: {sizes['train_size']:,} / "
            f"{sizes['validation_size']:,} / {sizes['test_size']:,}"
        )
    st.subheader("Goalkeeper vs outfield")
    _show_figure(lambda: plot_population_comparison(get_comparison(data)))


def _render_performance(data: EvaluationData, population: str | None) -> None:
    st.header("Performance")
    st.caption("Held-out test set. " + SPLIT_CAVEAT)
    total_rows = sum(
        get_metrics(data, name)["test_samples"] for name in _selected_populations(population)
    )
    max_points: int | None = None
    if total_rows > MIN_SCATTER_POINTS:
        scatter_points = st.slider(
            "Scatter points shown",
            min_value=MIN_SCATTER_POINTS,
            max_value=total_rows,
            value=min(DEFAULT_SCATTER_POINTS, total_rows),
            step=SCATTER_STEP,
            help="A fixed-seed random subsample, so the same points appear on every run.",
            key=f"scatter_points_{population}",
        )
        max_points = int(scatter_points) if isinstance(scatter_points, (int, float)) else None
    st.subheader("Actual vs predicted")
    _show_figure(
        lambda: plot_actual_vs_predicted(data.predictions, population, max_points=max_points)
    )
    st.subheader("Error distribution")
    _show_figure(lambda: plot_error_distribution(data.predictions, population))
    st.subheader("MAE by actual rating range")
    names = _selected_populations(population)
    for column, name in zip(st.columns(len(names)), names):
        with column:
            _show_figure(
                lambda name=name: plot_mae_by_rating_range(get_rating_ranges(data, name), name)
            )


def _render_training(data: EvaluationData, population: str | None) -> None:
    st.header("Training")
    st.caption(
        "Values below are the training history stored with each checkpoint and are validation "
        "metrics, not the test metrics shown elsewhere."
    )
    for name in _selected_populations(population):
        summary = get_training_summary(data, name)
        st.subheader(name.title())
        columns = st.columns(4)
        columns[0].metric("Epochs stored", f"{summary['epochs_stored']:,}")
        columns[1].metric("Best epoch (stored)", f"{summary['best_epoch']}")
        columns[2].metric("Best validation MAE (stored)", f"{summary['best_validation_mae']:.4f}")
        columns[3].metric(
            "Early-stopping flag (stored)",
            _format_flag(summary["stopped_early"]),
            help="Shown exactly as recorded in the evaluation artifact.",
        )
        _show_figure(
            lambda name=name: plot_training_history(get_training_history(data, name), name)
        )


def _render_error_analysis(data: EvaluationData, population: str | None) -> None:
    st.header("Error Analysis")
    st.caption("Error is predicted minus actual rating. " + SPLIT_CAVEAT)
    st.subheader("Error direction")
    for name in _selected_populations(population):
        metrics = get_metrics(data, name)
        st.markdown(f"**{name.title()}**")
        columns = st.columns(3)
        for column, label, key in (
            (columns[0], "Underpredicted (error < 0)", "underprediction"),
            (columns[1], "Overpredicted (error > 0)", "overprediction"),
            (columns[2], "Exactly zero error", "zero_error"),
        ):
            column.metric(label, f"{metrics[f'{key}_count']:,}")
            column.caption(f"{metrics[f'{key}_rate']:.2%} of test rows")
    st.subheader("Rating-range summary")
    st.caption("Bands are by actual rating. Bands with very few samples should not be generalized.")
    for name in _selected_populations(population):
        st.markdown(f"**{name.title()}**")
        st.dataframe(
            get_rating_ranges(data, name),
            hide_index=True,
            column_config={
                "rating_range": st.column_config.TextColumn("Rating range"),
                "samples": st.column_config.NumberColumn("Samples", format="%d"),
                "actual_mean": st.column_config.NumberColumn("Mean actual", format="%.2f"),
                "mean_error": st.column_config.NumberColumn("Mean error", format="%.3f"),
                "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
            },
        )
    st.subheader("Largest prediction errors")
    limit = st.slider("Errors shown", min_value=5, max_value=25, value=10, key="largest_limit")
    rows = filter_predictions(data.predictions, population)
    _show_figure(lambda: plot_largest_errors(rows, limit=limit))


def _render_test_predictions(data: EvaluationData, population: str | None) -> None:
    st.header("Test Predictions")
    st.caption(
        "Stored held-out test predictions from the V1 evaluation. This table is not a live "
        "predictor. Change the population in the sidebar."
    )
    bounds = get_prediction_ranges(data.predictions, population)
    rating_low, rating_high = bounds["actual_overall"]
    error_low, error_high = bounds["absolute_error"]
    scope = population or "both"
    left, middle, right = st.columns(3)
    rating_range = left.slider(
        "Actual rating range",
        min_value=rating_low,
        max_value=rating_high,
        value=(rating_low, rating_high),
        step=1.0,
        key=f"rating_range_{scope}",
    )
    error_range = middle.slider(
        "Absolute error range",
        min_value=error_low,
        max_value=error_high,
        value=(error_low, error_high),
        step=0.01,
        key=f"error_range_{scope}",
    )
    direction_label = right.selectbox("Error direction", list(DIRECTION_CHOICES), key="direction")
    player_text = st.text_input(
        "Player ID lookup", value="", placeholder="e.g. 158023", key="player_id"
    )
    try:
        rows = filter_predictions(
            data.predictions,
            population,
            rating_range=rating_range,
            error_range=error_range,
            direction=DIRECTION_CHOICES[str(direction_label)],
            player_id=player_text,
        )
    except VisualizationDataError as error:
        st.error(_safe_message(error))
        return
    total_rows = len(filter_predictions(data.predictions, population))
    st.write(
        f"**{len(rows):,}** of {total_rows:,} rows match, "
        "sorted by absolute error (largest first)."
    )
    if rows.empty:
        st.info("No rows match the current filters.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        height=420,
        column_config={
            "player_id": st.column_config.NumberColumn("Player ID", format="%d"),
            "actual_overall": st.column_config.NumberColumn("Actual", format="%.0f"),
            "predicted_overall": st.column_config.NumberColumn("Predicted", format="%.2f"),
            "error": st.column_config.NumberColumn("Error", format="%.3f"),
            "absolute_error": st.column_config.NumberColumn("Absolute error", format="%.3f"),
            "population": st.column_config.TextColumn("Population"),
        },
    )
    st.download_button(
        "Download filtered rows (CSV)",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="v1_test_predictions_filtered.csv",
        mime="text/csv",
    )


RENDERERS: dict[str, Callable[[EvaluationData, str | None], None]] = {
    "Overview": _render_overview,
    "Performance": _render_performance,
    "Training": _render_training,
    "Error Analysis": _render_error_analysis,
    "Test Predictions": _render_test_predictions,
}


def main() -> None:
    st.set_page_config(page_title="FIFA OVR V1 Evaluation", layout="wide")
    try:
        data = _load_data()
        section, population = _render_sidebar(data)
    except VisualizationDataError as error:
        st.title("FIFA Overall Rating: V1 Evaluation")
        st.error(
            "The evaluation artifacts could not be loaded. Expected "
            "`reports/generated/v1_final_evaluation.json` and "
            "`reports/generated/v1_test_predictions.csv`."
        )
        st.code(_safe_message(error))
        st.stop()
    try:
        RENDERERS[section](data, population)
    except VisualizationDataError as error:
        st.error(f"This section could not be displayed: {_safe_message(error)}")


main()
