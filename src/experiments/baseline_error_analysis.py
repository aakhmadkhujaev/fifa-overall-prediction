"""Evaluate and explain the committed FIFA baseline checkpoints."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.data.data_loader import load_dataset
from src.evaluation.error_analysis import calculate_error_direction
from src.features.feature_engineering import (
    get_goalkeeper_features,
    get_outfield_features,
    select_features,
    split_gk_and_outfield,
)
from src.preprocessing.preprocessor import Preprocessor
from src.training.model import FIFAOverallModel
from src.training.trainer import calculate_regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_error_analysis.md"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "baseline_error_analysis_figures"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEVICE = torch.device("cpu")
PathLike = Union[str, Path]


@dataclass
class PopulationAnalysis:
    """All tabular evidence generated for one player population."""

    name: str
    is_goalkeeper: bool
    test_features: pd.DataFrame
    test_targets: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float]
    observations: pd.DataFrame
    rating_ranges: pd.DataFrame
    large_errors: pd.DataFrame
    feature_associations: pd.DataFrame
    checkpoint_path: Path


def calculate_error_metrics(
    actual: Sequence[float] | pd.Series | np.ndarray,
    predictions: Sequence[float] | pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return regression and signed-error metrics for aligned values."""
    core_metrics = calculate_regression_metrics(
        torch.as_tensor(np.asarray(predictions, dtype=np.float32).copy()),
        torch.as_tensor(np.asarray(actual, dtype=np.float32).copy()),
    )
    directional_metrics = calculate_error_direction(actual, predictions)
    return {
        "mae": float(core_metrics["mae"]),
        "rmse": float(core_metrics["rmse"]),
        "r2": float(core_metrics["r2"]),
        "mean_error": float(directional_metrics["mean_error"]),
        "mean_absolute_error": float(core_metrics["mae"]),
        "test_samples": int(directional_metrics["test_samples"]),
        "underprediction_rate": float(directional_metrics["underprediction_rate"]),
        "overprediction_rate": float(directional_metrics["overprediction_rate"]),
        "zero_error_rate": float(directional_metrics["zero_error_rate"]),
    }


def load_baseline_model(checkpoint_path: PathLike, input_size: int) -> FIFAOverallModel:
    """Load a 5C checkpoint into the existing CPU model architecture."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=DEVICE, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    model = FIFAOverallModel(input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict_test_set(model: torch.nn.Module, features: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
    """Predict every row in a preprocessed test feature frame on CPU."""
    if features.empty:
        raise ValueError("Test features are empty.")
    feature_tensor = torch.as_tensor(features.to_numpy(copy=True), dtype=torch.float32)
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature_tensor), batch_size):
            output = model(feature_tensor[start:start + batch_size]).reshape(-1)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)


def _rating_ranges(actual: pd.Series, residuals: np.ndarray) -> pd.DataFrame:
    """Summarize stable, interpretable FIFA rating bands."""
    bins = [-np.inf, 59.999, 69.999, 79.999, 89.999, np.inf]
    labels = ["<60", "60-69", "70-79", "80-89", "90+"]
    frame = pd.DataFrame({"actual": actual.to_numpy(), "residual": residuals})
    frame["rating_range"] = pd.cut(frame["actual"], bins=bins, labels=labels, right=True)
    summary = frame.groupby("rating_range", observed=False).agg(
        samples=("actual", "size"),
        actual_mean=("actual", "mean"),
        mean_error=("residual", "mean"),
        mae=("residual", lambda values: np.abs(values).mean()),
        rmse=("residual", lambda values: np.sqrt(np.square(values).mean())),
    ).reset_index()
    return summary[summary["samples"] > 0].reset_index(drop=True)


def _large_errors(observations: pd.DataFrame, feature_names: Iterable[str], limit: int = 10) -> pd.DataFrame:
    """Return the largest errors with non-identifying, useful attributes."""
    useful_columns = ["actual", "prediction", "residual", "absolute_error"]
    useful_columns += [feature for feature in ("age", "height_cm", "weight_kg") if feature in feature_names]
    return observations.sort_values("absolute_error", ascending=False).loc[:, useful_columns].head(limit).reset_index(drop=True)


def _feature_associations(observations: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Measure Spearman association with signed and absolute error."""
    rows = []
    for feature in feature_names:
        if feature == "preferred_foot" or feature not in observations:
            continue
        numeric_feature = pd.to_numeric(observations[feature], errors="coerce")
        if numeric_feature.nunique(dropna=True) < 2:
            continue
        rows.append({
            "feature": feature,
            "signed_error_spearman": numeric_feature.corr(observations["residual"], method="spearman"),
            "absolute_error_spearman": numeric_feature.corr(observations["absolute_error"], method="spearman"),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["feature", "signed_error_spearman", "absolute_error_spearman"])
    return result.sort_values("absolute_error_spearman", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)


def analyze_population(
    population_df: pd.DataFrame,
    *,
    name: str,
    is_goalkeeper: bool,
    checkpoint_path: PathLike,
) -> PopulationAnalysis:
    """Recreate the 5C split, preprocess it, and analyze one checkpoint."""
    selected = select_features(population_df, is_goalkeeper=is_goalkeeper)
    preprocessor = Preprocessor(is_goalkeeper=is_goalkeeper)
    features, targets, _ = preprocessor.process(selected)
    model = load_baseline_model(checkpoint_path, input_size=len(features["train"].columns))
    predictions = predict_test_set(model, features["test"])
    actual = targets["test"].reset_index(drop=True)
    residuals = predictions - actual.to_numpy()
    observations = population_df.loc[features["test"].index].reset_index(drop=True).copy()
    observations["actual"] = actual.to_numpy()
    observations["prediction"] = predictions
    observations["residual"] = residuals
    observations["absolute_error"] = np.abs(residuals)
    feature_names = get_goalkeeper_features() if is_goalkeeper else get_outfield_features()
    return PopulationAnalysis(
        name=name,
        is_goalkeeper=is_goalkeeper,
        test_features=features["test"],
        test_targets=actual,
        predictions=predictions,
        metrics=calculate_error_metrics(actual, predictions),
        observations=observations,
        rating_ranges=_rating_ranges(actual, residuals),
        large_errors=_large_errors(observations, feature_names),
        feature_associations=_feature_associations(observations, feature_names),
        checkpoint_path=Path(checkpoint_path),
    )


def _save_plots(analysis: PopulationAnalysis, figures_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = analysis.name.lower()
    paths = {
        "actual_predicted": figures_dir / f"{prefix}_actual_vs_predicted.png",
        "residuals": figures_dir / f"{prefix}_residual_distribution.png",
        "ranges": figures_dir / f"{prefix}_error_by_rating_range.png",
    }
    plt.figure(figsize=(7, 5))
    plt.scatter(analysis.observations["actual"], analysis.observations["prediction"], s=5, alpha=0.25)
    limits = [analysis.observations["actual"].min(), analysis.observations["actual"].max()]
    plt.plot(limits, limits, color="black", linewidth=1)
    plt.xlabel("Actual overall")
    plt.ylabel("Predicted overall")
    plt.title(f"{analysis.name}: actual vs predicted")
    plt.tight_layout()
    plt.savefig(paths["actual_predicted"], dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(analysis.observations["residual"], bins=40, edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual (prediction - actual)")
    plt.ylabel("Test samples")
    plt.title(f"{analysis.name}: residual distribution")
    plt.tight_layout()
    plt.savefig(paths["residuals"], dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(analysis.rating_ranges["rating_range"].astype(str), analysis.rating_ranges["mae"])
    plt.xlabel("Actual overall range")
    plt.ylabel("MAE")
    plt.title(f"{analysis.name}: MAE by actual rating range")
    plt.tight_layout()
    plt.savefig(paths["ranges"], dpi=140)
    plt.close()
    return {key: str(path) for key, path in paths.items()}


def _format_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        if column in {"samples", "test_samples"}:
            formatted[column] = formatted[column].map(lambda value: f"{int(value)}" if pd.notna(value) else "")
        else:
            formatted[column] = formatted[column].map(lambda value: f"{value:.{float_digits}f}" if pd.notna(value) else "")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def _analysis_section(analysis: PopulationAnalysis, figure_paths: Mapping[str, str]) -> str:
    metrics = analysis.metrics
    return f"""## {analysis.name} analysis

Checkpoint: `{analysis.checkpoint_path.name}`. The checkpoint was loaded on CPU and evaluated on {metrics['test_samples']:,} test rows after the existing grouped split and train-fitted preprocessing.

### Test metrics

| Metric | Value |
| --- | ---: |
| MAE | {metrics['mae']:.6f} |
| RMSE | {metrics['rmse']:.6f} |
| R2 | {metrics['r2']:.6f} |
| Mean error (prediction - actual) | {metrics['mean_error']:.6f} |
| Mean absolute error | {metrics['mean_absolute_error']:.6f} |
| Underprediction rate | {metrics['underprediction_rate']:.2%} |
| Overprediction rate | {metrics['overprediction_rate']:.2%} |

### Rating ranges

Bins are fixed FIFA-style bands selected before looking at error values. Empty bands are omitted; counts therefore describe the actual test distribution.

{_format_table(analysis.rating_ranges)}

### Residuals and large errors

Residuals are defined as prediction minus actual. A positive mean error indicates overall overprediction; a negative value indicates overall underprediction. The largest absolute errors are shown without player names or IDs:

{_format_table(analysis.large_errors, 3)}

![{analysis.name} actual versus predicted]({figure_paths['actual_predicted']})
![{analysis.name} residual distribution]({figure_paths['residuals']})
![{analysis.name} error by rating range]({figure_paths['ranges']})

### Feature/error observations

The five largest absolute Spearman associations with absolute error are shown below. These are observational associations within this test set, not causal effects or evidence that a feature should be added, removed, or transformed.

{_format_table(analysis.feature_associations.head(5), 4)}

"""


def write_report(analyses: Sequence[PopulationAnalysis], report_path: PathLike, figure_paths: Mapping[str, Mapping[str, str]]) -> None:
    """Write the evidence-based Markdown report."""
    comparison = pd.DataFrame([
        {"population": analysis.name, **analysis.metrics}
        for analysis in analyses
    ])[['population', 'test_samples', 'mae', 'rmse', 'r2', 'mean_error', 'underprediction_rate', 'overprediction_rate']]
    report = f"""# Baseline Error Analysis

## Objective

This analysis characterizes the behavior and limitations of the existing 5C goalkeeper and outfield baseline models. It does not redesign, tune, retrain, or compare the populations as a ranking.

## Data/model sources

- Source data: the repository's raw male-player FIFA dataset.
- Features: the existing `select_features` definitions (12 goalkeeper features and 35 outfield features).
- Models: the existing CPU PyTorch checkpoints and `FIFAOverallModel` architecture.
- Split/preprocessing: the existing `Preprocessor.process` grouped split with `random_state=42`; imputation and scaling were fit on training rows and only transformed the validation and test rows.

## Evaluation methodology

Predictions were generated for every row in each complete test split. Residual means use `prediction - actual`; absolute error is `abs(residual)`. MAE, RMSE, R2, residual direction, fixed rating bands, large-error rows, and Spearman feature/error associations were computed from those predictions. No test values were used to fit preprocessing.

## Population comparison

This table is descriptive only. Differences reflect the two populations, their feature sets, and their test distributions; they do not establish that one model is preferable.

{_format_table(comparison, 6)}

{_analysis_section(analyses[0], figure_paths[analyses[0].name])}
{_analysis_section(analyses[1], figure_paths[analyses[1].name])}
## Limitations

- This is a single fixed grouped test split, so results describe this split rather than all possible future FIFA data.
- Correlation does not establish causation, feature importance, or a justified modeling change.
- Large-error rows identify where the baseline misses most; they do not by themselves explain why.
- The report does not claim that either population needs a particular model, feature, or hyperparameter change.

## Evidence-based findings

- The exact measured test metrics and residual direction are reported above for each population.
- Rating-range tables show whether observed error levels vary across actual-rating bands and include sample counts to make distribution support visible.
- Large-error tables and association tables identify concrete follow-up questions while preserving the distinction between observation and explanation.

## Implications for future work

Future milestones can use the reported rating bands, large-error cases, and strongest observed associations to define targeted hypotheses and additional validation. Any modeling change should be tested separately with leakage-safe splits and held-out evaluation; these findings alone do not select a redesign.
"""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")


def run_analysis(
    dataset_path: Optional[PathLike] = None,
    *,
    models_dir: PathLike = DEFAULT_MODELS_DIR,
    report_path: PathLike = DEFAULT_REPORT_PATH,
    figures_dir: PathLike = DEFAULT_FIGURES_DIR,
) -> Tuple[PopulationAnalysis, PopulationAnalysis]:
    """Run the complete baseline analysis without changing model artifacts."""
    dataset = load_dataset(dataset_path)
    goalkeeper_df, outfield_df = split_gk_and_outfield(dataset)
    models_path = Path(models_dir)
    analyses = (
        analyze_population(
            goalkeeper_df,
            name="Goalkeeper",
            is_goalkeeper=True,
            checkpoint_path=models_path / "fifa_overall_goalkeeper_baseline.pt",
        ),
        analyze_population(
            outfield_df,
            name="Outfield",
            is_goalkeeper=False,
            checkpoint_path=models_path / "fifa_overall_outfield_baseline.pt",
        ),
    )
    report_directory = Path(report_path).parent.resolve()
    figure_paths = {
        analysis.name: {
            key: os.path.relpath(path, report_directory).replace("\\", "/")
            for key, path in _save_plots(analysis, Path(figures_dir)).items()
        }
        for analysis in analyses
    }
    write_report(analyses, report_path, figure_paths)
    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()
    analyses = run_analysis(args.dataset, models_dir=args.models_dir, report_path=args.report, figures_dir=args.figures_dir)
    for analysis in analyses:
        print(f"{analysis.name}: {analysis.metrics}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()