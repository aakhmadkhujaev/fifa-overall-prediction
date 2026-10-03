import numpy as np
import pandas as pd
import torch

from src.experiments.baseline_error_analysis import (
    analyze_population,
    calculate_error_metrics,
    load_baseline_model,
    predict_test_set,
)
from src.features.feature_engineering import get_goalkeeper_features
from src.training.model import FIFAOverallModel


def test_calculate_error_metrics_reports_signed_and_directional_errors():
    metrics = calculate_error_metrics([60, 70, 80], [61, 68, 80])

    assert metrics["test_samples"] == 3
    assert np.isclose(metrics["mean_error"], -1 / 3)
    assert np.isclose(metrics["mean_absolute_error"], 1.0)
    assert np.isclose(metrics["underprediction_rate"], 1 / 3)
    assert np.isclose(metrics["overprediction_rate"], 1 / 3)


def test_temporary_checkpoint_loads_and_predicts_on_cpu(tmp_path):
    checkpoint_path = tmp_path / "baseline.pt"
    source_model = FIFAOverallModel(input_size=2)
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    loaded_model = load_baseline_model(checkpoint_path, input_size=2)
    predictions = predict_test_set(loaded_model, pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))

    assert predictions.shape == (2,)
    assert all(parameter.device.type == "cpu" for parameter in loaded_model.parameters())


def test_analyze_population_uses_test_rows_and_reports_rating_ranges(tmp_path):
    features = get_goalkeeper_features()
    rows = []
    for player_id in range(12):
        for record in range(2):
            row = {"player_id": player_id, "overall": 60 + player_id}
            row.update({feature: ("Right" if feature == "preferred_foot" else float(50 + player_id + record)) for feature in features})
            rows.append(row)
    data = pd.DataFrame(rows)
    checkpoint_path = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": FIFAOverallModel(input_size=len(features)).state_dict()}, checkpoint_path)

    analysis = analyze_population(data, name="Goalkeeper", is_goalkeeper=True, checkpoint_path=checkpoint_path)

    assert analysis.metrics["test_samples"] == len(analysis.observations)
    assert analysis.rating_ranges["samples"].sum() == analysis.metrics["test_samples"]
    assert {"residual", "absolute_error"}.issubset(analysis.observations.columns)