"""Evaluate official FIFA Overall V1 checkpoints on their held-out test splits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset
from src.evaluation.error_analysis import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    calculate_error_direction,
    largest_absolute_errors,
    summarize_rating_ranges,
)
from src.experiments.artifact import V1Artifact, load_v1_checkpoint
from src.features.feature_engineering import select_features, split_gk_and_outfield
from src.preprocessing.preprocessor import split_with_target_handling
from src.training.trainer import calculate_regression_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "male_players (legacy).csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "generated" / "v1_final_evaluation.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "generated" / "v1_test_predictions.csv"
INFERENCE_BATCH_SIZE = 2048
SPLIT_TEST_SIZE = 0.15
SPLIT_VAL_SIZE = 0.15
SPLIT_RANDOM_STATE = 42
PathLike = Union[str, Path]


@dataclass(frozen=True)
class PopulationEvaluation:
    """Structured evidence for one population's held-out test evaluation."""

    population: str
    checkpoint_path: str
    split_sizes: Dict[str, int]
    metrics: Dict[str, float | int]
    error_analysis: Dict[str, Any]
    training_history: Dict[str, Any]
    checkpoint_metadata: Dict[str, Any]


class EvaluationError(RuntimeError):
    """Raised when an official artifact cannot be evaluated safely."""


def predict_in_batches(
    artifact: V1Artifact,
    features: pd.DataFrame,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """Run CPU inference without changing the restored model or preprocessor."""
    if features.empty:
        raise EvaluationError("Test features are empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    transformed = artifact.preprocessor.transform_features(features)
    tensor = torch.as_tensor(transformed.to_numpy(copy=True), dtype=torch.float32)
    predictions = []
    artifact.model.to(torch.device("cpu"))
    artifact.model.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            output = artifact.model(tensor[start:start + batch_size]).reshape(-1)
            predictions.append(output.cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise EvaluationError("Predictions contain NaN or infinite values.")
    return result


def _checkpoint_metadata_summary(artifact: V1Artifact) -> Dict[str, Any]:
    metadata = artifact.checkpoint["checkpoint_metadata"]
    return {
        "checkpoint_schema_version": metadata["checkpoint_schema_version"],
        "population": metadata["population"],
        "model_class": metadata["model_class"],
        "experiment_config": dict(metadata["experiment_config"]),
        "model_config": dict(metadata["model_config"]),
        "input_feature_count": metadata["input_feature_count"],
        "feature_names": list(metadata["feature_names"]),
        "target_col": metadata["target_col"],
        "group_col": metadata["group_col"],
        "preprocessing_schema_version": metadata["preprocessing_schema_version"],
    }


def _history_summary(artifact: V1Artifact) -> Dict[str, Any]:
    checkpoint = artifact.checkpoint
    history = checkpoint.get("history")
    if not isinstance(history, Mapping):
        raise EvaluationError("Checkpoint training history is missing.")
    return {
        "train_loss": list(history.get("train_loss", [])),
        "validation_loss": list(history.get("val_loss", [])),
        "validation_mae": list(history.get("val_mae", [])),
        "validation_rmse": list(history.get("val_rmse", [])),
        "validation_r2": list(history.get("val_r2", [])),
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_mae": checkpoint["best_val_mae"],
        "epochs_completed": history.get("epochs_completed"),
        "stopped_early": history.get("stopped_early"),
    }


def evaluate_population(
    population_df: pd.DataFrame,
    *,
    population: str,
    checkpoint_path: PathLike,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> tuple[PopulationEvaluation, pd.DataFrame]:
    """Evaluate one official V1 artifact on its deterministic held-out split."""
    artifact = load_v1_checkpoint(checkpoint_path)
    metadata = artifact.checkpoint["checkpoint_metadata"]
    if metadata["population"] != population:
        raise EvaluationError(
            f"Checkpoint population {metadata['population']!r} does not match {population!r}."
        )

    selected = select_features(
        population_df, is_goalkeeper=(population == "goalkeeper")
    )
    train_df, validation_df, test_df = split_with_target_handling(
        selected,
        target_col=artifact.preprocessor.target_col,
        group_col=artifact.preprocessor.group_col,
        test_size=SPLIT_TEST_SIZE,
        val_size=SPLIT_VAL_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )
    raw_test_features = test_df.drop(
        columns=[artifact.preprocessor.target_col, artifact.preprocessor.group_col]
    )
    expected_features = list(artifact.preprocessor.feature_names)
    if list(raw_test_features.columns) != expected_features:
        raise EvaluationError("Test feature order does not match checkpoint metadata.")

    predictions = predict_in_batches(artifact, raw_test_features, batch_size=batch_size)
    prediction_frame = build_prediction_frame(
        test_df[artifact.preprocessor.group_col],
        test_df[artifact.preprocessor.target_col].to_numpy(),
        predictions,
        population,
    )
    actual_tensor = torch.as_tensor(
        prediction_frame["actual_overall"].to_numpy(copy=True), dtype=torch.float32
    )
    predicted_tensor = torch.as_tensor(
        prediction_frame["predicted_overall"].to_numpy(copy=True), dtype=torch.float32
    )
    metrics = {
        **calculate_regression_metrics(predicted_tensor, actual_tensor),
        **calculate_error_direction(
            prediction_frame["actual_overall"], prediction_frame["predicted_overall"]
        ),
        "test_samples": len(prediction_frame),
    }
    rating_ranges = summarize_rating_ranges(prediction_frame)
    largest_errors = largest_absolute_errors(prediction_frame)
    error_analysis = {
        "underprediction_count": metrics["underprediction_count"],
        "overprediction_count": metrics["overprediction_count"],
        "zero_error_count": metrics["zero_error_count"],
        "underprediction_rate": metrics["underprediction_rate"],
        "overprediction_rate": metrics["overprediction_rate"],
        "zero_error_rate": metrics["zero_error_rate"],
        "rating_ranges": rating_ranges.to_dict(orient="records"),
        "largest_absolute_errors": largest_errors.to_dict(orient="records"),
    }
    population_result = PopulationEvaluation(
        population=population,
        checkpoint_path=str(checkpoint_path),
        split_sizes={
            "train_size": len(train_df),
            "validation_size": len(validation_df),
            "test_size": len(test_df),
        },
        metrics=metrics,
        error_analysis=error_analysis,
        training_history=_history_summary(artifact),
        checkpoint_metadata=_checkpoint_metadata_summary(artifact),
    )
    return population_result, prediction_frame[list(PREDICTION_COLUMNS)].copy()


def _population_to_dict(result: PopulationEvaluation) -> Dict[str, Any]:
    return asdict(result)


def evaluate_final_v1(
    dataset_path: Optional[PathLike] = None,
    *,
    models_dir: PathLike = DEFAULT_MODELS_DIR,
    results_path: PathLike = DEFAULT_RESULTS_PATH,
    predictions_path: PathLike = DEFAULT_PREDICTIONS_PATH,
) -> Dict[str, Any]:
    """Evaluate both official V1 checkpoints and write structured outputs."""
    dataset_file = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    dataset = load_dataset(dataset_file)
    validation = validate_dataset(dataset)
    if not validation["target_exists"]:
        raise EvaluationError("Dataset does not contain the overall target column.")
    goalkeeper_df, outfield_df = split_gk_and_outfield(dataset)
    checkpoint_paths = {
        "goalkeeper": Path(models_dir) / "fifa_overall_goalkeeper_v1.pt",
        "outfield": Path(models_dir) / "fifa_overall_outfield_v1.pt",
    }

    started = time.perf_counter()
    evaluations = []
    prediction_frames = []
    for population, population_df in (
        ("goalkeeper", goalkeeper_df),
        ("outfield", outfield_df),
    ):
        result, frame = evaluate_population(
            population_df,
            population=population,
            checkpoint_path=checkpoint_paths[population],
        )
        evaluations.append(result)
        prediction_frames.append(frame)

    prediction_output = pd.concat(prediction_frames, ignore_index=True)
    prediction_output_path = Path(predictions_path)
    prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(prediction_output_path, index=False)

    comparison = {
        result.population: {
            key: result.metrics[key] for key in ("mae", "rmse", "r2", "test_samples")
        }
        for result in evaluations
    }
    output = {
        "schema_version": 1,
        "evaluation": {
            "dataset_path": str(dataset_file),
            "dataset_rows": len(dataset),
            "validation": validation,
            "test_size": SPLIT_TEST_SIZE,
            "validation_size": SPLIT_VAL_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "prediction_csv": str(prediction_output_path),
            "runtime_seconds": time.perf_counter() - started,
        },
        "populations": {
            result.population: _population_to_dict(result) for result in evaluations
        },
        "comparison": comparison,
    }
    results_output_path = Path(results_path)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    evaluate_final_v1()
