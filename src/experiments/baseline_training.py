"""Run the first reproducible FIFA baseline training experiment."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset
from src.features.feature_engineering import select_features, split_gk_and_outfield
from src.preprocessing.preprocessor import Preprocessor
from src.training.dataloader import create_dataloaders
from src.training.model import FIFAOverallModel
from src.training.trainer import (
    calculate_regression_metrics,
    set_seed,
    train_model,
)


RANDOM_SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 30
PATIENCE = 7
DEVICE = "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "baseline_training_results.json"

PathLike = Union[str, Path]


@dataclass
class ModelResults:
    """Metrics and metadata for one trained player-group model."""

    model_type: str
    train_samples: int
    validation_samples: int
    test_samples: int
    input_features: int
    best_epoch: int
    best_validation_mae: float
    best_validation_rmse: float
    best_validation_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    training_seconds: float
    checkpoint_path: str


class BaselineExperimentError(RuntimeError):
    """Raised when the baseline experiment cannot prepare valid input data."""


def _evaluate_test_set(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Evaluate the restored best-validation model on the complete test set."""
    predictions = []
    targets = []
    model.eval()

    with torch.no_grad():
        for features, batch_targets in test_loader:
            predictions.append(model(features.to(DEVICE)).cpu())
            targets.append(batch_targets.cpu())

    if not predictions:
        raise BaselineExperimentError("Test DataLoader is empty.")

    return calculate_regression_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _metric_at_best_epoch(history: Dict[str, Any], metric_name: str) -> float:
    """Return a validation metric from the epoch selected by validation MAE."""
    best_epoch = history.get("best_epoch")
    metric_values = history.get(metric_name)
    if best_epoch is None or not metric_values:
        raise BaselineExperimentError(
            f"Training history does not contain a best epoch and {metric_name}."
        )
    return float(metric_values[int(best_epoch) - 1])


def _run_player_group(
    group_df: pd.DataFrame,
    *,
    model_type: str,
    is_goalkeeper: bool,
    checkpoint_path: Path,
) -> ModelResults:
    """Prepare, train, and evaluate one independent player-group model."""
    selected_df = select_features(group_df, is_goalkeeper=is_goalkeeper)
    preprocessor = Preprocessor(is_goalkeeper=is_goalkeeper)
    features, targets, split_stats = preprocessor.process(selected_df)
    dataloaders = create_dataloaders(
        features,
        targets,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    input_features = int(features["train"].shape[1])
    set_seed(RANDOM_SEED)
    model = FIFAOverallModel(input_size=input_features)
    checkpoint_metadata = {
        "checkpoint_schema_version": 2,
        "model_type": model_type,
        "input_features": input_features,
        "feature_names": list(features["train"].columns),
        "preprocessor_state": preprocessor.get_state(),
    }

    print(f"Training {model_type} model...")
    training_started = time.perf_counter()
    history = train_model(
        model,
        dataloaders["train"],
        dataloaders["val"],
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=DEVICE,
        patience=PATIENCE,
        checkpoint_path=checkpoint_path,
        seed=RANDOM_SEED,
        checkpoint_metadata=checkpoint_metadata,
    )
    training_seconds = time.perf_counter() - training_started

    test_metrics = _evaluate_test_set(model, dataloaders["test"])
    best_epoch = history.get("best_epoch")
    if best_epoch is None:
        raise BaselineExperimentError("Training history does not contain a best epoch.")

    return ModelResults(
        model_type=model_type,
        train_samples=int(split_stats["train_size"]),
        validation_samples=int(split_stats["val_size"]),
        test_samples=int(split_stats["test_size"]),
        input_features=input_features,
        best_epoch=int(best_epoch),
        best_validation_mae=float(history["best_val_mae"]),
        best_validation_rmse=_metric_at_best_epoch(history, "val_rmse"),
        best_validation_r2=_metric_at_best_epoch(history, "val_r2"),
        test_mae=test_metrics["mae"],
        test_rmse=test_metrics["rmse"],
        test_r2=test_metrics["r2"],
        training_seconds=training_seconds,
        checkpoint_path=str(checkpoint_path),
    )


def run_baseline_experiment(
    dataset_path: Optional[PathLike] = None,
    *,
    models_dir: PathLike = DEFAULT_MODELS_DIR,
    results_path: PathLike = DEFAULT_RESULTS_PATH,
) -> Dict[str, Any]:
    """Run and persist the goalkeeper and outfield baseline experiments."""
    set_seed(RANDOM_SEED)
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    validation_report = validate_dataset(dataset)
    if not validation_report["target_exists"]:
        raise BaselineExperimentError("Dataset does not contain the overall target column.")
    invalid_target_issues = [
        issue
        for issue in validation_report["target_issues"]
        if issue != "Target column contains missing values."
    ]
    if invalid_target_issues:
        raise BaselineExperimentError(
            "Dataset contains invalid target values: "
            + "; ".join(invalid_target_issues)
        )
    print(f"Dataset loaded: {len(dataset)} rows")

    print("Preparing goalkeeper data...")
    goalkeeper_df, outfield_df = split_gk_and_outfield(dataset)
    models_path = Path(models_dir)
    goalkeeper_checkpoint = models_path / "fifa_overall_goalkeeper_baseline.pt"
    outfield_checkpoint = models_path / "fifa_overall_outfield_baseline.pt"

    total_started = time.perf_counter()
    goalkeeper_results = _run_player_group(
        goalkeeper_df,
        model_type="goalkeeper",
        is_goalkeeper=True,
        checkpoint_path=goalkeeper_checkpoint,
    )

    print("Preparing outfield data...")
    outfield_results = _run_player_group(
        outfield_df,
        model_type="outfield",
        is_goalkeeper=False,
        checkpoint_path=outfield_checkpoint,
    )
    total_seconds = time.perf_counter() - total_started

    results = {
        "configuration": {
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "device": DEVICE,
            "hidden_sizes": [64, 32],
            "num_workers": 0,
        },
        "dataset": {
            "path": str(dataset_path or PROJECT_ROOT / "data" / "raw" / "male_players (legacy).csv"),
            "rows": len(dataset),
            "columns": len(dataset.columns),
            "goalkeeper_samples": len(goalkeeper_df),
            "outfield_samples": len(outfield_df),
            "validation": validation_report,
        },
        "goalkeeper": asdict(goalkeeper_results),
        "outfield": asdict(outfield_results),
        "training_duration_seconds": total_seconds,
    }

    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Baseline experiment complete.")
    return results


def main() -> None:
    """Run the baseline experiment with repository-default paths."""
    run_baseline_experiment()


if __name__ == "__main__":
    main()
