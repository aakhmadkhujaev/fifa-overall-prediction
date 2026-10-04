from dataclasses import replace
from typing import Optional

import pandas as pd
import pytest

from src.experiments.batch_size_search import (
    DEFAULT_BATCH_SIZES,
    BatchSizeSearchConfig,
    BatchSizeSearcher,
    generate_batch_size_candidates,
    make_milestone_6c_search_config,
)
from src.experiments.experiment import ExperimentConfig, ExperimentResult, Population


def make_config(population: Population = "outfield") -> BatchSizeSearchConfig:
    return make_milestone_6c_search_config("batch-size-smoke", population)


def make_result(
    config: ExperimentConfig,
    validation_mae: float,
    test_mae: Optional[float],
) -> ExperimentResult:
    return ExperimentResult(
        experiment_name=config.experiment_name,
        population=config.population,
        configuration=config,
        best_epoch=3,
        best_validation_mae=validation_mae,
        validation_mae=validation_mae,
        validation_rmse=validation_mae + 0.1,
        validation_r2=0.5,
        test_mae=test_mae,
        test_rmse=test_mae + 0.1 if test_mae is not None else None,
        test_r2=0.4 if test_mae is not None else None,
        training_duration_seconds=0.01,
    )


class FakeRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        calls: list[tuple[ExperimentConfig, pd.DataFrame, bool]],
        validation_by_batch_size: dict[int, float],
        test_by_batch_size: dict[int, float],
    ) -> None:
        self.config = config
        self.calls = calls
        self.validation_by_batch_size = validation_by_batch_size
        self.test_by_batch_size = test_by_batch_size

    def run(
        self,
        population_data: pd.DataFrame,
        *,
        evaluate_test: bool = True,
    ) -> ExperimentResult:
        self.calls.append((self.config, population_data, evaluate_test))
        batch_size = self.config.batch_size
        validation_mae = self.validation_by_batch_size[batch_size]
        test_mae = self.test_by_batch_size[batch_size] if evaluate_test else None
        return make_result(self.config, validation_mae, test_mae)


def test_default_candidates_are_exact_and_deterministic():
    config = make_config()

    assert generate_batch_size_candidates(config) == (64, 128, 256, 512, 1024)
    assert generate_batch_size_candidates(config) == generate_batch_size_candidates(config)
    assert DEFAULT_BATCH_SIZES == (64, 128, 256, 512, 1024)


@pytest.mark.parametrize("invalid_sizes", [(), (0, 128), (-1,), (64, 64), (64, 128.0)])
def test_invalid_batch_sizes_are_rejected(invalid_sizes):
    with pytest.raises(ValueError, match="batch_sizes"):
        BatchSizeSearchConfig(
            "invalid",
            "outfield",
            ExperimentConfig(
                "fixed",
                "outfield",
                learning_rate=0.0017320508075688787,
            ),
            invalid_sizes,
        )


@pytest.mark.parametrize(
    "population, learning_rate",
    [("goalkeeper", 0.01), ("outfield", 0.0017320508075688787)],
)
def test_population_uses_fixed_learning_rate(population: Population, learning_rate: float):
    config = make_config(population)

    assert config.fixed_config.learning_rate == learning_rate


def test_non_search_fixed_settings_are_preserved():
    config = make_config()
    fixed = config.fixed_config

    assert fixed.random_seed == 42
    assert fixed.epochs == 30
    assert fixed.patience == 7
    assert fixed.device == "cpu"
    assert fixed.num_workers == 0
    assert (fixed.hidden_size1, fixed.hidden_size2) == (64, 32)


@pytest.mark.parametrize("population", ["goalkeeper", "outfield"])
def test_search_uses_runner_protocol_and_validation_only_selection(population: Population):
    config = make_config(population)
    calls: list[tuple[ExperimentConfig, pd.DataFrame, bool]] = []
    validation = {64: 0.8, 128: 0.6, 256: 0.7, 512: 0.9, 1024: 0.75}
    test = {64: 0.1, 128: 0.9, 256: 0.2, 512: 0.3, 1024: 0.4}

    result = BatchSizeSearcher(
        config,
        runner_factory=lambda candidate_config: FakeRunner(
            candidate_config, calls, validation, test
        ),
    ).run(pd.DataFrame({"synthetic": [1]}))

    candidate_calls = [call for call in calls if not call[2]]
    final_calls = [call for call in calls if call[2]]
    assert [config.batch_size for config, _, _ in candidate_calls] == list(DEFAULT_BATCH_SIZES)
    assert all(evaluate_test is False for _, _, evaluate_test in candidate_calls)
    assert all(candidate.test_mae is None for candidate in result.candidate_results)
    assert result.selected_batch_size == 128
    assert result.selected_validation_mae == 0.6
    assert result.best_experiment_result.test_mae == 0.9
    assert len(final_calls) == 1
    assert final_calls[0][0] is result.best_experiment_result.configuration
    assert final_calls[0][0].batch_size == 128
    assert final_calls[0][2] is True
    assert result.total_trials == len(DEFAULT_BATCH_SIZES)


def test_fixed_config_mismatch_is_rejected():
    with pytest.raises(ValueError, match="learning rate"):
        BatchSizeSearchConfig(
            "invalid-rate",
            "goalkeeper",
            replace(ExperimentConfig("fixed", "goalkeeper"), learning_rate=0.001),
        )