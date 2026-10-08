import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.visualization import data as data_module
from tests.test_visualization import GK_ROWS, OF_ROWS, make_predictions, make_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
SECTIONS = ("Overview", "Performance", "Training", "Error Analysis", "Test Predictions")
POPULATIONS = ("Both", "Goalkeeper", "Outfield")
TOTAL_ROWS = GK_ROWS + OF_ROWS
FAKE_PATH_MARKERS = ("does\\not\\exist", "C:\\")


@pytest.fixture(autouse=True)
def clear_cached_data():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture()
def results() -> dict:
    return make_results(make_predictions())


def write_artifacts(tmp_path: Path, results: dict) -> tuple[Path, Path]:
    results_path = tmp_path / "evaluation.json"
    predictions_path = tmp_path / "predictions.csv"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    make_predictions().to_csv(predictions_path, index=False)
    return results_path, predictions_path


def point_app_at(monkeypatch, results_path: Path, predictions_path: Path) -> None:
    monkeypatch.setattr(data_module, "DEFAULT_RESULTS_PATH", results_path)
    monkeypatch.setattr(data_module, "DEFAULT_PREDICTIONS_PATH", predictions_path)


@pytest.fixture()
def app(monkeypatch, tmp_path, results) -> AppTest:
    point_app_at(monkeypatch, *write_artifacts(tmp_path, results))
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def navigate(app: AppTest, section: str, population: str = "Both") -> AppTest:
    app.sidebar.radio(key="section").set_value(section)
    app.sidebar.radio(key="population").set_value(population)
    return app.run()


def visible_text(app: AppTest) -> str:
    parts: list[str] = []
    for collection in (
        app.title, app.header, app.subheader, app.markdown, app.caption, app.text, app.code,
        app.error, app.info, app.warning,
    ):
        parts.extend(str(element.value) for element in collection)
    for metric in app.metric:
        parts.extend([metric.label, str(metric.value)])
    return "\n".join(parts)


def metric_values(app: AppTest) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for metric in app.metric:
        values.setdefault(metric.label, []).append(str(metric.value))
    return values


def rows_matching_text(app: AppTest) -> str:
    return next(
        str(element.value) for element in app.markdown if "rows match" in str(element.value)
    )


# ------------------------------------------------------------------ rendering


def test_app_starts_on_overview(app):
    assert not app.exception
    assert [radio.label for radio in app.sidebar.radio] == ["Section", "Population"]
    assert app.sidebar.radio(key="section").value == "Overview"
    assert app.sidebar.radio(key="population").value == "Both"
    assert app.header[0].value == "Overview"


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("population", POPULATIONS)
def test_every_section_renders_for_every_population(app, section, population):
    navigate(app, section, population)

    assert not app.exception
    assert not app.error
    assert app.header[0].value == section


def test_navigation_renders_only_the_selected_section(app):
    navigate(app, "Training")
    assert [header.value for header in app.header] == ["Training"]
    navigate(app, "Test Predictions")
    assert [header.value for header in app.header] == ["Test Predictions"]


def test_sidebar_shows_run_summary_without_paths(app):
    sidebar_text = " ".join(str(caption.value) for caption in app.sidebar.caption)
    assert "1,000" in sidebar_text
    assert "42" in sidebar_text
    assert not any(marker in sidebar_text for marker in FAKE_PATH_MARKERS)


@pytest.mark.parametrize("section", SECTIONS)
def test_no_json_paths_are_displayed(app, section):
    navigate(app, section)
    text = visible_text(app)
    assert not any(marker in text for marker in FAKE_PATH_MARKERS)
    assert "players.csv" not in text


# ------------------------------------------------------------------- overview


def test_overview_both_populations_shows_stored_metrics(app, results):
    values = metric_values(navigate(app, "Overview", "Both"))

    assert values["Test rows"] == [f"{GK_ROWS:,}", f"{OF_ROWS:,}"]
    gk_mae = results["populations"]["goalkeeper"]["metrics"]["mae"]
    assert values["MAE"][0] == f"{gk_mae:.3f}"
    assert len(values["R\u00b2"]) == 2
    assert app.subheader[-1].value == "Goalkeeper vs outfield"


def test_overview_single_population(app):
    values = metric_values(navigate(app, "Overview", "Outfield"))
    assert values["Test rows"] == [f"{OF_ROWS:,}"]
    assert len(values["MAE"]) == 1


# ------------------------------------------------------------------- training


@pytest.mark.parametrize(
    ("stored", "expected"), [(True, "True"), (False, "False"), (None, "Not recorded")]
)
def test_training_shows_stored_early_stopping_flag_as_stored(
    monkeypatch, tmp_path, results, stored, expected
):
    results["populations"]["goalkeeper"]["training_history"]["stopped_early"] = stored
    point_app_at(monkeypatch, *write_artifacts(tmp_path, results))
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    values = metric_values(navigate(app, "Training", "Goalkeeper"))

    assert values["Early-stopping flag (stored)"] == [expected]
    assert "stored" in values["Best epoch (stored)"] or values["Best epoch (stored)"]


def test_training_does_not_infer_early_stopping_from_epoch_counts(
    monkeypatch, tmp_path, results
):
    history = results["populations"]["outfield"]["training_history"]
    history["stopped_early"] = False
    history["epochs_completed"] = 3  # fewer epochs than a configured 30; must not matter
    point_app_at(monkeypatch, *write_artifacts(tmp_path, results))
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    values = metric_values(navigate(app, "Training", "Outfield"))

    assert values["Early-stopping flag (stored)"] == ["False"]


def test_training_shows_best_epoch_and_validation_mae(app, results):
    values = metric_values(navigate(app, "Training", "Goalkeeper"))
    history = results["populations"]["goalkeeper"]["training_history"]
    assert values["Best epoch (stored)"] == [str(history["best_epoch"])]
    assert values["Best validation MAE (stored)"] == [f"{history['best_validation_mae']:.4f}"]
    assert values["Epochs stored"] == [str(len(history["train_loss"]))]


# ------------------------------------------------------------- error analysis


def test_error_analysis_shows_direction_counts_and_tables(app, results):
    navigate(app, "Error Analysis", "Goalkeeper")
    metrics = results["populations"]["goalkeeper"]["metrics"]
    values = metric_values(app)

    assert values["Underpredicted (error < 0)"] == [f"{metrics['underprediction_count']:,}"]
    assert values["Overpredicted (error > 0)"] == [f"{metrics['overprediction_count']:,}"]
    assert values["Exactly zero error"] == [f"{metrics['zero_error_count']:,}"]
    assert len(app.dataframe) == 1
    assert {"rating_range", "samples", "mae"} <= set(app.dataframe[0].value.columns)


def test_error_analysis_limit_slider_runs(app):
    navigate(app, "Error Analysis", "Both")
    app.slider(key="largest_limit").set_value(5).run()
    assert not app.exception
    assert not app.error


# ----------------------------------------------------------- test predictions


def test_test_predictions_shows_all_rows_by_default(app):
    navigate(app, "Test Predictions", "Both")

    assert f"**{TOTAL_ROWS:,}** of {TOTAL_ROWS:,} rows match" in rows_matching_text(app)
    table = app.dataframe[0].value
    assert len(table) == TOTAL_ROWS
    assert list(table["absolute_error"]) == sorted(table["absolute_error"], reverse=True)


def test_test_predictions_population_changes_scope(app):
    navigate(app, "Test Predictions", "Goalkeeper")
    assert f"**{GK_ROWS:,}** of {GK_ROWS:,} rows match" in rows_matching_text(app)
    assert set(app.dataframe[0].value["population"]) == {"goalkeeper"}

    navigate(app, "Test Predictions", "Outfield")
    assert f"**{OF_ROWS:,}** of {OF_ROWS:,} rows match" in rows_matching_text(app)


@pytest.mark.parametrize(
    ("label", "column_check"),
    [
        ("Underpredicted (error < 0)", lambda frame: (frame["error"] < 0).all()),
        ("Overpredicted (error > 0)", lambda frame: (frame["error"] > 0).all()),
    ],
)
def test_test_predictions_direction_filter(app, label, column_check):
    navigate(app, "Test Predictions", "Both")
    app.selectbox(key="direction").set_value(label).run()

    table = app.dataframe[0].value
    assert 0 < len(table) < TOTAL_ROWS
    assert column_check(table)
    assert f"**{len(table):,}** of {TOTAL_ROWS:,}" in rows_matching_text(app)


def test_test_predictions_player_id_lookup(app):
    navigate(app, "Test Predictions", "Both")
    app.text_input(key="player_id").set_value("1003").run()

    table = app.dataframe[0].value
    assert list(table["player_id"]) == [1003]
    assert f"**1** of {TOTAL_ROWS:,}" in rows_matching_text(app)


def test_test_predictions_unknown_player_shows_info_not_table(app):
    navigate(app, "Test Predictions", "Both")
    app.text_input(key="player_id").set_value("999999").run()

    assert not app.exception
    assert not app.dataframe
    assert any("No rows match" in str(info.value) for info in app.info)
    assert f"**0** of {TOTAL_ROWS:,}" in rows_matching_text(app)


def test_test_predictions_invalid_player_id_shows_message(app):
    navigate(app, "Test Predictions", "Both")
    app.text_input(key="player_id").set_value("not-a-number").run()

    assert not app.exception
    assert any("player_id" in str(error.value) for error in app.error)


def test_test_predictions_absolute_error_range_filter(app):
    navigate(app, "Test Predictions", "Both")
    slider = app.slider(key="error_range_both")
    low, high = slider.value
    slider.set_range(low, 0.6).run()

    table = app.dataframe[0].value
    assert 0 < len(table) < TOTAL_ROWS
    assert (table["absolute_error"] <= 0.6 + 1e-9).all()


def test_test_predictions_rating_range_filter(app):
    navigate(app, "Test Predictions", "Both")
    slider = app.slider(key="rating_range_both")
    _, high = slider.value
    slider.set_range(60.0, high).run()

    table = app.dataframe[0].value
    assert (table["actual_overall"] >= 60.0).all()
    assert 0 < len(table) < TOTAL_ROWS


def test_test_predictions_provides_csv_download(app):
    navigate(app, "Test Predictions", "Both")
    buttons = app.get("download_button")
    assert len(buttons) == 1
    assert "CSV" in buttons[0].proto.label


def test_test_predictions_has_no_download_when_nothing_matches(app):
    navigate(app, "Test Predictions", "Both")
    app.text_input(key="player_id").set_value("999999").run()
    assert not app.get("download_button")


# ---------------------------------------------------------- missing artifacts


def test_missing_artifacts_show_a_clear_error(monkeypatch):
    missing_dir = PROJECT_ROOT / "reports" / "generated"
    point_app_at(monkeypatch, missing_dir / "absent.json", missing_dir / "absent.csv")

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not app.exception
    assert any("could not be loaded" in str(error.value) for error in app.error)
    assert not app.header  # no section content rendered
    text = visible_text(app) + "\n".join(str(code.value) for code in app.code)
    assert "not found" in text
    assert str(PROJECT_ROOT) not in text


def test_invalid_artifacts_show_a_clear_error(monkeypatch, tmp_path, results):
    results["schema_version"] = 99
    point_app_at(monkeypatch, *write_artifacts(tmp_path, results))

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not app.exception
    assert any("could not be loaded" in str(error.value) for error in app.error)
    assert "schema_version" in "\n".join(str(code.value) for code in app.code)


def test_loader_failure_is_not_cached(monkeypatch, tmp_path, results):
    missing = PROJECT_ROOT / "reports" / "generated"
    point_app_at(monkeypatch, missing / "absent.json", missing / "absent.csv")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert app.error

    point_app_at(monkeypatch, *write_artifacts(tmp_path, results))
    app.run()

    assert not app.error
    assert app.header[0].value == "Overview"


# ------------------------------------------------------ committed artifacts


COMMITTED = (
    data_module.DEFAULT_RESULTS_PATH.exists() and data_module.DEFAULT_PREDICTIONS_PATH.exists()
)


@pytest.mark.skipif(not COMMITTED, reason="committed evaluation artifacts are not present")
class TestCommittedArtifacts:
    def test_all_sections_render(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        assert not app.exception
        for section in SECTIONS:
            for population in POPULATIONS:
                navigate(app, section, population)
                assert not app.exception, (section, population)
                assert not app.error, (section, population)

    def test_scatter_slider_is_bounded_and_runs(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        navigate(app, "Performance", "Both")
        slider = app.slider(key="scatter_points_None")
        assert slider.min == 500
        assert slider.max == 24371
        assert slider.value == 5000
        slider.set_value(500).run()
        assert not app.exception

    def test_all_prediction_rows_are_available_in_the_explorer(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        navigate(app, "Test Predictions", "Both")
        assert "**24,371** of 24,371 rows match" in rows_matching_text(app)
        assert isinstance(app.dataframe[0].value, pd.DataFrame)


# ----------------------------------------------------- static / import isolation


FORBIDDEN_IMPORTS = (
    "torch",
    "seaborn",
    "joblib",
    "sklearn",
    "src.training",
    "src.experiments",
    "src.data.data_loader",
    "src.evaluation",
    "src.prediction",
)
ALLOWED_IMPORTS = {
    "__future__",
    "collections.abc",
    "streamlit",
    "matplotlib.figure",
    "src.visualization",
    "src.visualization.data",
}
FORBIDDEN_CALLS = {
    "open", "read_csv", "read_json", "read_text", "write_text", "to_json", "load",
    "loads", "savefig", "show", "cache_data", "train", "fit", "predict",
}


def _app_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def _app_imports() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_app_imports_only_streamlit_and_the_visualization_layer():
    imports = _app_imports()
    assert imports <= ALLOWED_IMPORTS, imports - ALLOWED_IMPORTS
    for module in imports:
        for forbidden in FORBIDDEN_IMPORTS:
            assert module != forbidden and not module.startswith(forbidden + ".")


def test_app_does_no_file_io_training_or_figure_caching():
    called = set()
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    assert called.isdisjoint(FORBIDDEN_CALLS), called & FORBIDDEN_CALLS


def test_app_contains_no_raw_json_schema_keys():
    # "populations" and "schema_version" are also keys of the get_run_summary() accessor
    # output, so they are legitimate in app.py and are not checked here.
    raw_keys = {
        "error_analysis",
        "checkpoint_metadata",
        "checkpoint_path",
        "training_history",
        "evaluation",
        "comparison",
        "split_sizes",
        "largest_absolute_errors",
    }
    literals = {
        node.value
        for node in ast.walk(_app_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert literals.isdisjoint(raw_keys), literals & raw_keys
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "results"
        for node in ast.walk(_app_tree())
    )


def test_app_caches_only_the_loaded_evaluation_data():
    decorated = [
        node.name
        for node in ast.walk(_app_tree())
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    assert decorated == ["_load_data"]


def test_running_the_app_does_not_import_torch_or_ml_modules():
    code = (
        "import sys\n"
        "from streamlit.testing.v1 import AppTest\n"
        f"app = AppTest.from_file({str(APP_PATH)!r}, default_timeout=120).run()\n"
        "bad = [m for m in ('torch', 'sklearn', 'joblib', 'seaborn', 'src.training', "
        "'src.experiments', 'src.data.data_loader', 'src.evaluation.v1_evaluation', "
        "'src.prediction') if m in sys.modules]\n"
        "print('EXC' if app.exception else 'OK', ','.join(bad))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip().splitlines()[-1] == "OK"
