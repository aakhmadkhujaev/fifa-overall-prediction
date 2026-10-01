import pytest
import torch
from torch import nn

from src.training.model import FIFAOverallModel


def test_model_can_be_instantiated_and_is_a_torch_module():
    model = FIFAOverallModel(input_size=10)

    assert isinstance(model, nn.Module)


def test_model_accepts_configured_input_feature_count():
    model = FIFAOverallModel(input_size=7)

    assert model.input_size == 7
    assert model.network[0].in_features == 7


def test_forward_pass_returns_float32_output_with_one_value_per_row():
    model = FIFAOverallModel(input_size=10)
    inputs = torch.randn(8, 10, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (8, 1)
    assert outputs.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_different_batch_sizes_work(batch_size):
    model = FIFAOverallModel(input_size=5)
    inputs = torch.randn(batch_size, 5, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (batch_size, 1)


@pytest.mark.parametrize("input_size", [3, 12])
def test_different_input_feature_counts_work(input_size):
    model = FIFAOverallModel(input_size=input_size)
    inputs = torch.randn(6, input_size, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (6, 1)


def test_invalid_feature_dimensions_fail_clearly():
    model = FIFAOverallModel(input_size=10)
    inputs = torch.randn(4, 9, dtype=torch.float32)

    with pytest.raises(ValueError, match="Expected input tensor with 10 features"):
        model(inputs)


def test_model_contains_exactly_one_final_output_neuron():
    model = FIFAOverallModel(input_size=10)

    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]

    assert linear_layers[-1].out_features == 1
    assert sum(layer.out_features for layer in linear_layers[-1:]) == 1


def test_trainable_parameter_count_is_calculated():
    model = FIFAOverallModel(input_size=10)
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    expected_count = (10 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
    assert trainable_parameter_count == expected_count
