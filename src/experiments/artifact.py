"""Validation and reconstruction helpers for official V1 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Mapping, Union

import torch

from src.preprocessing.preprocessor import Preprocessor
from src.training.model import FIFAOverallModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = {2}
VALID_POPULATIONS = {"goalkeeper", "outfield"}
SUPPORTED_MODEL_CLASSES = {"FIFAOverallModel"}
PathLike = Union[str, Path]


def portable_path(path: PathLike) -> str:
    """Return a machine-independent path string for generated artifacts.

    Paths inside the project become project-relative POSIX paths. Paths outside it
    are reduced to the file name so no local directory layout is recorded.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


@dataclass(frozen=True)
class V1Artifact:
    """A validated V1 checkpoint and its reconstructed inference components."""

    checkpoint: Mapping[str, Any]
    model: FIFAOverallModel
    preprocessor: Preprocessor


def load_v1_checkpoint(checkpoint_path: PathLike) -> V1Artifact:
    """Validate and reconstruct a CPU-loadable V1 checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("checkpoint_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata is missing.")
    required_metadata = (
        "checkpoint_schema_version",
        "population",
        "model_class",
        "experiment_config",
        "input_features",
        "input_feature_count",
        "feature_names",
        "target_col",
        "group_col",
        "preprocessing_schema_version",
        "preprocessing_state",
        "model_config",
    )
    missing_metadata = [field for field in required_metadata if field not in metadata]
    if missing_metadata:
        raise ValueError(f"Checkpoint metadata is missing: {missing_metadata}")

    required_checkpoint = (
        "model_state_dict",
        "optimizer_state_dict",
        "best_epoch",
        "best_val_mae",
    )
    missing_checkpoint = [field for field in required_checkpoint if field not in checkpoint]
    if missing_checkpoint:
        raise ValueError(f"Checkpoint is missing: {missing_checkpoint}")

    schema_version = metadata["checkpoint_schema_version"]
    if schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported checkpoint schema version: {schema_version}")
    if metadata["population"] not in VALID_POPULATIONS:
        raise ValueError("Checkpoint population is invalid.")
    if metadata["model_class"] not in SUPPORTED_MODEL_CLASSES:
        raise ValueError(f"Unsupported model class: {metadata['model_class']}")
    if not isinstance(metadata["experiment_config"], Mapping):
        raise ValueError("Checkpoint experiment configuration is invalid.")
    if not isinstance(metadata["target_col"], str) or not metadata["target_col"]:
        raise ValueError("Checkpoint target_col is invalid.")
    if not isinstance(metadata["group_col"], str) or not metadata["group_col"]:
        raise ValueError("Checkpoint group_col is invalid.")
    if not isinstance(metadata["preprocessing_schema_version"], int):
        raise ValueError("Checkpoint preprocessing schema version is invalid.")

    feature_names = metadata["feature_names"]
    if not isinstance(feature_names, list) or not feature_names or any(
        not isinstance(name, str) for name in feature_names
    ):
        raise ValueError("Checkpoint feature names are missing or unordered.")
    if not isinstance(metadata["input_features"], int) or metadata["input_features"] <= 0:
        raise ValueError("Checkpoint input_features is invalid.")
    if metadata["input_feature_count"] != metadata["input_features"]:
        raise ValueError("Checkpoint input feature counts do not match.")
    model_config = metadata["model_config"]
    if not isinstance(model_config, Mapping) or model_config.get("model_class") != metadata["model_class"]:
        raise ValueError("Checkpoint model configuration is missing.")
    if not isinstance(metadata["preprocessing_state"], Mapping):
        raise ValueError("Checkpoint preprocessing state is missing.")
    if metadata["preprocessing_state"].get("schema_version") != metadata["preprocessing_schema_version"]:
        raise ValueError("Checkpoint preprocessing schema versions do not match.")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("Checkpoint model state is missing.")
    if not isinstance(checkpoint["optimizer_state_dict"], Mapping):
        raise ValueError("Checkpoint optimizer state is missing.")
    if isinstance(checkpoint["best_epoch"], bool) or not isinstance(checkpoint["best_epoch"], int):
        raise ValueError("Checkpoint best_epoch is invalid.")
    if not isinstance(checkpoint["best_val_mae"], Real):
        raise ValueError("Checkpoint best_val_mae is invalid.")

    input_size = int(model_config.get("input_size", 0))
    if input_size != len(feature_names):
        raise ValueError("Checkpoint input feature count does not match feature names.")
    model = FIFAOverallModel(
        input_size=input_size,
        hidden_size1=int(model_config.get("hidden_size1", 0)),
        hidden_size2=int(model_config.get("hidden_size2", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    preprocessor = Preprocessor.from_state(metadata["preprocessing_state"])
    if list(preprocessor.feature_names) != feature_names:
        raise ValueError("Preprocessor feature order does not match checkpoint metadata.")
    model.eval()
    return V1Artifact(checkpoint=checkpoint, model=model, preprocessor=preprocessor)
