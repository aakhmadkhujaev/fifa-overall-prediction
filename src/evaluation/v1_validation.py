"""Consistency checks for the generated V1 evaluation artifacts.

The checks read only committed-style JSON/CSV files; they never load checkpoints,
the raw dataset, or torch. Run ``python -m src.evaluation.v1_validation`` after
regenerating the artifacts (see the README).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import PREDICTION_COLUMNS
from src.features.feature_engineering import get_goalkeeper_features, get_outfield_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_ROOT / "reports" / "generated"
DEFAULT_EVALUATION_PATH = GENERATED_DIR / "v1_final_evaluation.json"
DEFAULT_PREDICTIONS_PATH = GENERATED_DIR / "v1_test_predictions.csv"
DEFAULT_TRAINING_RECORD_PATH = GENERATED_DIR / "v1_final_results.json"
POPULATIONS = ("goalkeeper", "outfield")
HISTORY_SERIES = (
    "train_loss", "validation_loss", "validation_mae", "validation_rmse", "validation_r2"
)
TEST_METRICS = ("mae", "rmse", "r2")
ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2}[^\\/\s])")
PathLike = Union[str, Path]


def _load_json(path: PathLike) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strings(value: Any, location: str = "$"):
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _strings(child, f"{location}[{index}]")


def find_absolute_paths(document: Any, name: str) -> list[str]:
    """Return a problem for every string value that looks like an absolute path."""
    return [
        f"{name}: absolute path at {location}"
        for location, text in _strings(document)
        if ABSOLUTE_PATH.match(text)
    ]


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _check_history(population: str, history: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    lengths = {key: len(history.get(key, [])) for key in HISTORY_SERIES}
    epochs = history.get("epochs_completed")
    if len(set(lengths.values())) != 1 or epochs != lengths["train_loss"]:
        problems.append(
            f"{population}: history lengths {lengths} do not match epochs_completed={epochs}"
        )
        return problems
    best_epoch = history.get("best_epoch")
    if (
        not isinstance(best_epoch, int)
        or not isinstance(epochs, int)
        or not 1 <= best_epoch <= epochs
    ):
        return problems + [f"{population}: best_epoch {best_epoch!r} is outside 1..{epochs!r}"]
    if not _close(
        history["validation_mae"][best_epoch - 1], history["best_validation_mae"], 1e-9
    ):
        problems.append(f"{population}: validation_mae at best_epoch differs from best_validation_mae")
    stopped = record.get("early_stopping", {}).get("stopped_early")
    for label, stored, recorded in (
        ("epochs_completed", epochs, record.get("epochs_completed")),
        ("stopped_early", history.get("stopped_early"), stopped),
        ("best_epoch", best_epoch, record.get("best_epoch")),
    ):
        if stored != recorded:
            problems.append(
                f"{population}: {label} is {stored!r} in the evaluation history "
                f"but {recorded!r} in the training record"
            )
    return problems


def _check_predictions(
    predictions: pd.DataFrame, evaluation: Mapping[str, Any]
) -> list[str]:
    if list(predictions.columns) != list(PREDICTION_COLUMNS):
        return [f"predictions: columns {list(predictions.columns)} != {list(PREDICTION_COLUMNS)}"]
    problems: list[str] = []
    numeric = predictions[list(PREDICTION_COLUMNS[:-1])].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        problems.append("predictions: NaN or infinite values present")
    error = predictions["predicted_overall"] - predictions["actual_overall"]
    if not np.allclose(error, predictions["error"], atol=1e-6):
        problems.append("predictions: error != predicted_overall - actual_overall")
    if not np.allclose(error.abs(), predictions["absolute_error"], atol=1e-6):
        problems.append("predictions: absolute_error != |error|")
    ids: dict[str, set] = {}
    for population in POPULATIONS:
        rows = predictions[predictions["population"] == population]
        expected = evaluation["populations"][population]["metrics"]["test_samples"]
        if len(rows) != expected:
            problems.append(f"{population}: {len(rows)} prediction rows, expected {expected}")
        ids[population] = set(rows["player_id"])
    if ids["goalkeeper"] & ids["outfield"]:
        problems.append("predictions: goalkeeper and outfield player_id sets overlap")
    unknown = set(predictions["population"]) - set(POPULATIONS)
    if unknown:
        problems.append(f"predictions: unknown populations {sorted(unknown)}")
    return problems


def validate_v1_artifacts(
    evaluation_path: PathLike = DEFAULT_EVALUATION_PATH,
    predictions_path: PathLike = DEFAULT_PREDICTIONS_PATH,
    training_record_path: PathLike = DEFAULT_TRAINING_RECORD_PATH,
    *,
    reference_record_path: Optional[PathLike] = None,
    tolerance: float = 1e-9,
) -> list[str]:
    """Return a list of problems found in the generated V1 artifacts (empty means valid).

    ``reference_record_path`` may point at a copy of the training record from before
    regeneration; predictive metrics must then match it within ``tolerance``.
    """
    evaluation = _load_json(evaluation_path)
    record = _load_json(training_record_path)
    problems = find_absolute_paths(evaluation, "evaluation JSON")
    problems += find_absolute_paths(record, "training record JSON")
    reference = _load_json(reference_record_path) if reference_record_path else None
    expected_features = {
        "goalkeeper": get_goalkeeper_features(),
        "outfield": get_outfield_features(),
    }
    for population in POPULATIONS:
        entry = evaluation["populations"][population]
        population_record = record[population]
        problems += _check_history(population, entry["training_history"], population_record)
        for metric in TEST_METRICS:
            if not _close(entry["metrics"][metric], population_record["test"][metric], 1e-6):
                problems.append(
                    f"{population}: evaluated test {metric} differs from the training record"
                )
            if reference is not None and not _close(
                population_record["test"][metric], reference[population]["test"][metric], tolerance
            ):
                problems.append(f"{population}: test {metric} changed from the reference record")
        if reference is not None:
            for key in ("best_epoch", "best_validation_mae"):
                if not _close(population_record[key], reference[population][key], tolerance):
                    problems.append(f"{population}: {key} changed from the reference record")
        metadata = entry["checkpoint_metadata"]
        if metadata["feature_names"] != expected_features[population]:
            problems.append(f"{population}: checkpoint feature order differs from the feature list")
        if metadata["input_feature_count"] != len(expected_features[population]):
            problems.append(f"{population}: input_feature_count does not match the feature list")
    predictions = pd.read_csv(predictions_path)
    problems += _check_predictions(predictions, evaluation)
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    reference = args[0] if args else None
    problems = validate_v1_artifacts(reference_record_path=reference)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("V1 artifacts are consistent." if not problems else f"{len(problems)} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
