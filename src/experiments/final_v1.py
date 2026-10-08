"""Orchestration for the two official FIFA V1 production experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset
from src.experiments.artifact import portable_path
from src.experiments.experiment import ExperimentConfig, ExperimentRunner
from src.features.feature_engineering import split_gk_and_outfield

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "male_players (legacy).csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "generated" / "v1_final_results.json"

PathLike = Union[str, Path]


def final_v1_configs() -> tuple[ExperimentConfig, ExperimentConfig]:
    """Return the fixed goalkeeper and outfield V1 configurations."""
    return (
        ExperimentConfig(
            experiment_name="fifa_overall_goalkeeper_v1",
            population="goalkeeper",
            learning_rate=0.01,
            batch_size=128,
            epochs=30,
            patience=7,
            random_seed=42,
            device="cpu",
            num_workers=0,
            hidden_size1=64,
            hidden_size2=32,
        ),
        ExperimentConfig(
            experiment_name="fifa_overall_outfield_v1",
            population="outfield",
            learning_rate=0.0017320508075688787,
            batch_size=256,
            epochs=30,
            patience=7,
            random_seed=42,
            device="cpu",
            num_workers=0,
            hidden_size1=64,
            hidden_size2=32,
        ),
    )


def _validate_dataset_or_raise(dataset: Any) -> Dict[str, Any]:
    report = validate_dataset(dataset)
    if not report["target_exists"]:
        raise ValueError("Dataset does not contain the overall target column.")
    invalid_target_issues = [
        issue for issue in report["target_issues"]
        if issue != "Target column contains missing values."
    ]
    if invalid_target_issues:
        raise ValueError(
            "Dataset contains invalid target values: " + "; ".join(invalid_target_issues)
        )
    return report


def _result_record(result: Any, config: ExperimentConfig, checkpoint_path: Path) -> Dict[str, Any]:
    return {
        "configuration": asdict(config),
        "checkpoint_path": portable_path(checkpoint_path),
        "feature_names": list(result.feature_names),
        "feature_count": result.input_features,
        "best_epoch": result.best_epoch,
        "best_validation_mae": result.best_validation_mae,
        "validation": {
            "mae": result.validation_mae,
            "rmse": result.validation_rmse,
            "r2": result.validation_r2,
        },
        "test": {
            "mae": result.test_mae,
            "rmse": result.test_rmse,
            "r2": result.test_r2,
        },
        "training_duration_seconds": result.training_duration_seconds,
        "runtime_seconds": result.training_duration_seconds,
        "split_sizes": result.split_sizes,
        "epochs_completed": result.epochs_completed,
        "early_stopping": {
            "stopped_early": result.stopped_early,
            "patience": config.patience,
        },
    }


def run_final_v1_experiment(
    dataset_path: Optional[PathLike] = None,
    *,
    models_dir: PathLike = DEFAULT_MODELS_DIR,
    results_path: PathLike = DEFAULT_RESULTS_PATH,
) -> Dict[str, Any]:
    """Train exactly one fixed V1 model for each player population."""
    dataset_file = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    dataset = load_dataset(dataset_file)
    validation = _validate_dataset_or_raise(dataset)
    goalkeeper_data, outfield_data = split_gk_and_outfield(dataset)
    population_data = {
        "goalkeeper": goalkeeper_data,
        "outfield": outfield_data,
    }
    checkpoint_paths = {
        "goalkeeper": Path(models_dir) / "fifa_overall_goalkeeper_v1.pt",
        "outfield": Path(models_dir) / "fifa_overall_outfield_v1.pt",
    }

    started = time.perf_counter()
    results: Dict[str, Any] = {}
    for config in final_v1_configs():
        result = ExperimentRunner(config).run(
            population_data[config.population],
            checkpoint_path=checkpoint_paths[config.population],
        )
        results[config.population] = _result_record(
            result, config, checkpoint_paths[config.population]
        )

    output = {
        "schema_version": 1,
        "dataset": {
            "path": portable_path(dataset_file),
            "rows": len(dataset),
            "population_rows": {
                "goalkeeper": len(goalkeeper_data),
                "outfield": len(outfield_data),
            },
            "validation": validation,
        },
        "goalkeeper": results["goalkeeper"],
        "outfield": results["outfield"],
        "runtime_seconds": time.perf_counter() - started,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    run_final_v1_experiment()
