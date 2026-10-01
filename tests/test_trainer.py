import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.model import FIFAOverallModel
from src.training.trainer import (
    calculate_regression_metrics,
    set_seed,
    train_model,
)


class FixedOutputModel(nn.Module):
    def __init__(self, output_value=0.0):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.output_value = output_value

    def forward(self, features):
        output = torch.full(
            (features.shape[0], 1),
            self.output_value,
            dtype=features.dtype,
            device=features.device,
        )
        return output + (self.dummy * 0.0)


class NonFinitePredictionModel(FixedOutputModel):
    def forward(self, features):
        output = torch.full(
            (features.shape[0], 1),
            float('nan'),
            dtype=features.dtype,
            device=features.device,
        )
        return output + (self.dummy * 0.0)


def make_loaders(feature_count=2, sample_count=32, batch_size=8):
    torch.manual_seed(7)
    features = torch.randn(sample_count, feature_count, dtype=torch.float32)
    targets = (features.sum(dim=1, keepdim=True) * 0.5) + 1.0
    train_dataset = TensorDataset(features[:24], targets[:24])
    validation_dataset = TensorDataset(features[24:], targets[24:])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False),
    )


def test_training_run_completes_and_records_expected_history():
    train_loader, validation_loader = make_loaders()
    model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)

    history = train_model(
        model,
        train_loader,
        validation_loader,
        epochs=3,
        patience=None,
    )

    assert history['epochs_completed'] == 3
    assert len(history['train_loss']) == 3
    assert len(history['val_loss']) == 3
    assert len(history['val_mae']) == 3
    assert len(history['val_rmse']) == 3
    assert len(history['val_r2']) == 3
    assert history['stopped_early'] is False


def test_validation_metrics_are_finite():
    train_loader, validation_loader = make_loaders()
    model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)

    history = train_model(model, train_loader, validation_loader, epochs=2)

    for metric_name in ('train_loss', 'val_loss', 'val_mae', 'val_rmse', 'val_r2'):
        assert all(torch.isfinite(torch.tensor(value)) for value in history[metric_name])


def test_model_parameters_change_after_training():
    train_loader, validation_loader = make_loaders()
    model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    train_model(model, train_loader, validation_loader, epochs=2)

    assert any(
        not torch.equal(initial, current.detach())
        for initial, current in zip(initial_parameters, model.parameters())
    )


def test_best_model_tracking_and_checkpoint_loading(tmp_path):
    train_loader, validation_loader = make_loaders()
    model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)
    checkpoint_path = tmp_path / 'best_model.pt'

    history = train_model(
        model,
        train_loader,
        validation_loader,
        epochs=4,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    restored_model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)
    restored_model.load_state_dict(checkpoint['model_state_dict'])

    assert checkpoint_path.exists()
    assert checkpoint['best_val_mae'] == history['best_val_mae']
    assert checkpoint['best_epoch'] == history['best_epoch']
    assert checkpoint['training_config']['learning_rate'] == 1e-3
    for name, parameter in restored_model.state_dict().items():
        assert torch.equal(parameter, history['best_state_dict'][name])


def test_training_restores_best_validation_state():
    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, features):
            return self.bias.expand(features.shape[0], 1)

    train_loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 1, dtype=torch.float32),
            torch.ones(1, 1, dtype=torch.float32),
        ),
        batch_size=1,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 1, dtype=torch.float32),
            torch.tensor([[0.01]], dtype=torch.float32),
        ),
        batch_size=1,
    )
    model = ScalarModel()

    history = train_model(
        model,
        train_loader,
        validation_loader,
        learning_rate=0.01,
        epochs=3,
    )

    assert history['best_epoch'] == 1
    assert history['val_mae'][1] > history['val_mae'][0]
    assert model.bias.item() == pytest.approx(0.01, abs=1e-6)


def test_early_stopping_uses_validation_mae():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, features):
            return self.bias.expand(features.shape[0], 1)

    train_features = torch.zeros(8, 2, dtype=torch.float32)
    train_targets = torch.zeros(8, 1, dtype=torch.float32)
    validation_features = torch.zeros(4, 2, dtype=torch.float32)
    validation_targets = torch.ones(4, 1, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=4)
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets), batch_size=4
    )

    history = train_model(
        ConstantModel(),
        train_loader,
        validation_loader,
        epochs=20,
        patience=2,
    )

    assert history['stopped_early'] is True
    assert history['epochs_completed'] == 3
    assert history['best_epoch'] == 1


def test_cpu_training_keeps_model_on_cpu():
    train_loader, validation_loader = make_loaders()
    model = FIFAOverallModel(input_size=2, hidden_size1=8, hidden_size2=4)

    train_model(model, train_loader, validation_loader, epochs=1, device='cpu')

    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())


@pytest.mark.parametrize('invalid_feature', [float('nan'), float('inf')])
def test_non_finite_features_are_rejected(invalid_feature):
    train_features = torch.tensor([[0.0, invalid_feature], [1.0, 2.0]])
    train_targets = torch.ones(2, 1, dtype=torch.float32)
    validation_features = torch.zeros(2, 2, dtype=torch.float32)
    validation_targets = torch.ones(2, 1, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=2)
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets), batch_size=2
    )

    with pytest.raises(ValueError, match='input features contain NaN or infinite values'):
        train_model(
            FIFAOverallModel(input_size=2),
            train_loader,
            validation_loader,
            epochs=1,
        )


@pytest.mark.parametrize('invalid_target', [float('nan'), float('inf')])
def test_non_finite_targets_are_rejected(invalid_target):
    train_features = torch.zeros(2, 2, dtype=torch.float32)
    train_targets = torch.tensor([[invalid_target], [1.0]])
    validation_features = torch.zeros(2, 2, dtype=torch.float32)
    validation_targets = torch.ones(2, 1, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=2)
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets), batch_size=2
    )

    with pytest.raises(ValueError, match='targets contain NaN or infinite values'):
        train_model(
            FIFAOverallModel(input_size=2),
            train_loader,
            validation_loader,
            epochs=1,
        )


def test_non_finite_predictions_are_rejected():
    features = torch.zeros(2, 2, dtype=torch.float32)
    targets = torch.ones(2, 1, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    validation_loader = DataLoader(TensorDataset(features, targets), batch_size=2)

    with pytest.raises(ValueError, match='model predictions contain NaN or infinite values'):
        train_model(
            NonFinitePredictionModel(),
            train_loader,
            validation_loader,
            epochs=1,
        )


def test_training_loss_is_sample_weighted_for_unequal_batches():
    train_features = torch.zeros(3, 1, dtype=torch.float32)
    train_targets = torch.tensor([[0.0], [2.0], [4.0]])
    validation_features = torch.zeros(1, 1, dtype=torch.float32)
    validation_targets = torch.zeros(1, 1, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=2, shuffle=False
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets), batch_size=1
    )

    history = train_model(
        FixedOutputModel(),
        train_loader,
        validation_loader,
        epochs=1,
    )

    expected_sample_weighted_loss = (0.0 + 4.0 + 16.0) / 3.0
    incorrect_batch_average = ((0.0 + 4.0) / 2.0 + 16.0) / 2.0
    assert history['train_loss'][0] == pytest.approx(expected_sample_weighted_loss)
    assert history['train_loss'][0] != pytest.approx(incorrect_batch_average)


def test_validation_metrics_use_complete_epoch_predictions():
    train_features = torch.zeros(1, 1, dtype=torch.float32)
    train_targets = torch.zeros(1, 1, dtype=torch.float32)
    validation_features = torch.zeros(4, 1, dtype=torch.float32)
    validation_targets = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=1)
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_targets), batch_size=3, shuffle=False
    )

    history = train_model(
        FixedOutputModel(),
        train_loader,
        validation_loader,
        epochs=1,
    )

    target_mean = validation_targets.mean()
    residual_sum_squares = torch.sum(torch.square(validation_targets))
    total_sum_squares = torch.sum(torch.square(validation_targets - target_mean))
    expected_mae = torch.abs(validation_targets).mean().item()
    expected_rmse = torch.sqrt(torch.square(validation_targets).mean()).item()
    expected_r2 = 1.0 - (residual_sum_squares / total_sum_squares).item()

    assert history['val_mae'][0] == pytest.approx(expected_mae)
    assert history['val_rmse'][0] == pytest.approx(expected_rmse)
    assert history['val_r2'][0] == pytest.approx(expected_r2)


def test_regression_metrics_handle_zero_target_variance():
    targets = torch.ones(4, 1, dtype=torch.float32)

    perfect_metrics = calculate_regression_metrics(targets, targets)
    incorrect_metrics = calculate_regression_metrics(torch.zeros_like(targets), targets)

    assert perfect_metrics == {'mae': 0.0, 'rmse': 0.0, 'r2': 1.0}
    assert incorrect_metrics['mae'] == 1.0
    assert incorrect_metrics['rmse'] == 1.0
    assert incorrect_metrics['r2'] == 0.0


def test_set_seed_makes_torch_random_values_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.randn(4))
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.randn(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
