"""Reusable configuration, execution, and result types for FIFA experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Union

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.features.feature_engineering import select_features
from src.preprocessing.preprocessor import Preprocessor
from src.training.dataloader import create_dataloaders
from src.training.model import FIFAOverallModel
from src.training.trainer import calculate_regression_metrics, set_seed, train_model


Population = Literal["goalkeeper", "outfield"]
PathLike = Union[str, Path]
DataLoaderFactory = Callable[..., Dict[str, DataLoader]]
ModelFactory = Callable[..., nn.Module]
TrainFunction = Callable[..., Dict[str, Any]]


class PreprocessorProtocol(Protocol):
    """Minimal preprocessing interface required by the experiment runner."""

    def process(
        self, data: pd.DataFrame
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, Any]]:
        ...


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings for one independent population experiment."""

    experiment_name: str
    population: Population
    random_seed: int = 42
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 30
    patience: Optional[int] = 7
    device: str = "cpu"
    num_workers: int = 0
    hidden_size1: int = 64
    hidden_size2: int = 32

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")
        if self.population not in {"goalkeeper", "outfield"}:
            raise ValueError("population must be 'goalkeeper' or 'outfield'.")
        for field_name in (
            "random_seed", "batch_size", "epochs", "num_workers", "hidden_size1", "hidden_size2"
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and greater than zero.")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if self.patience is not None and (
            isinstance(self.patience, bool) or not isinstance(self.patience, int)
        ):
            raise ValueError("patience must be an integer or None.")
        if self.patience is not None and self.patience < 0:
            raise ValueError("patience must be non-negative or None.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.hidden_size1 <= 0 or self.hidden_size2 <= 0:
            raise ValueError("Hidden layer sizes must be greater than zero.")
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid device: {self.device!r}.") from error


@dataclass(frozen=True)
class ExperimentResult:
    """Metrics and configuration captured from one completed experiment."""

    experiment_name: str
    population: Population
    configuration: ExperimentConfig
    best_epoch: int
    best_validation_mae: float
    validation_mae: float
    validation_rmse: float
    validation_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    training_duration_seconds: float


class ExperimentRunner:
    """Orchestrate preprocessing, model construction, training, and evaluation."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        preprocessor_factory: Callable[..., PreprocessorProtocol] = Preprocessor,
        dataloader_factory: DataLoaderFactory = create_dataloaders,
        model_factory: ModelFactory = FIFAOverallModel,
        train_function: TrainFunction = train_model,
    ) -> None:
        self.config = config
        self._preprocessor_factory = preprocessor_factory
        self._dataloader_factory = dataloader_factory
        self._model_factory = model_factory
        self._train_function = train_function

    def run(
        self,
        population_data: pd.DataFrame,
        *,
        checkpoint_path: Optional[PathLike] = None,
    ) -> ExperimentResult:
        """Run an experiment while keeping test evaluation after model selection."""
        is_goalkeeper = self.config.population == "goalkeeper"
        selected_data = select_features(population_data, is_goalkeeper=is_goalkeeper)
        preprocessor = self._preprocessor_factory(is_goalkeeper=is_goalkeeper)
        features, targets, _ = preprocessor.process(selected_data)
        dataloaders = self._dataloader_factory(
            features,
            targets,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        self._require_loaders(dataloaders)

        set_seed(self.config.random_seed)
        model = self._model_factory(
            input_size=int(features["train"].shape[1]),
            hidden_size1=self.config.hidden_size1,
            hidden_size2=self.config.hidden_size2,
        )

        started = time.perf_counter()
        history = self._train_function(
            model,
            dataloaders["train"],
            dataloaders["val"],
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            device=self.config.device,
            patience=self.config.patience,
            seed=self.config.random_seed,
            checkpoint_path=checkpoint_path,
        )
        duration = time.perf_counter() - started

        best_epoch = history.get("best_epoch")
        if best_epoch is None:
            raise ValueError("Training history does not contain a best epoch.")
        validation_metrics = self._history_metrics_at_epoch(history, int(best_epoch))
        test_metrics = self._evaluate(model, dataloaders["test"])

        return ExperimentResult(
            experiment_name=self.config.experiment_name,
            population=self.config.population,
            configuration=self.config,
            best_epoch=int(best_epoch),
            best_validation_mae=float(history["best_val_mae"]),
            validation_mae=validation_metrics["mae"],
            validation_rmse=validation_metrics["rmse"],
            validation_r2=validation_metrics["r2"],
            test_mae=test_metrics["mae"],
            test_rmse=test_metrics["rmse"],
            test_r2=test_metrics["r2"],
            training_duration_seconds=duration,
        )

    @staticmethod
    def _require_loaders(dataloaders: Dict[str, DataLoader]) -> None:
        missing = {name for name in ("train", "val", "test") if name not in dataloaders}
        if missing:
            raise ValueError(f"Dataloaders are missing required splits: {sorted(missing)}.")

    @staticmethod
    def _history_metrics_at_epoch(history: Dict[str, Any], epoch: int) -> Dict[str, float]:
        index = epoch - 1
        try:
            return {
                "mae": float(history["val_mae"][index]),
                "rmse": float(history["val_rmse"][index]),
                "r2": float(history["val_r2"][index]),
            }
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("Training history lacks metrics for the best epoch.") from error

    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        predictions = []
        targets = []
        device = torch.device(self.config.device)
        with torch.no_grad():
            for features, batch_targets in data_loader:
                predictions.append(model(features.to(device)).cpu())
                targets.append(batch_targets.cpu())
        if not predictions:
            raise ValueError("Test DataLoader is empty.")
        return calculate_regression_metrics(
            torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
        )