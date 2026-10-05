from typing import Optional

import pandas as pd
import pytest

from src.experiments.experiment import ExperimentConfig, ExperimentResult, Population
from src.experiments.training_duration import (
    make_milestone_6d_configs,
    run_training_duration_experiments,
)


def make_result(config: ExperimentConfig) -> ExperimentResult:
    return ExperimentResult(
        experiment_name=config.experiment_name,
        population=config.population,
        configuration=config,
        best_epoch=20,
        best_validation_mae=0.5,
        validation_mae=0.5,
        validation_rmse=0.6,
        validation_r2=0.7,
        test_mae=0.51,
        test_rmse=0.61,
        test_r2=0.69,
        training_duration_seconds=1.0,
    )


class FakeRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        calls: list[tuple[ExperimentConfig, pd.DataFrame, bool]],
        result: ExperimentResult,
    ) -> None:
        self.config = config
        self.calls = calls
        self.result = result

    def run(
        self,
        population_data: pd.DataFrame,
        *,
        evaluate_test: bool = True,
    ) -> ExperimentResult:
        self.calls.append((self.config, population_data, evaluate_test))
        return self.result


def test_milestone_6d_configs_use_optimized_population_settings():
    configs = make_milestone_6d_configs()

    assert configs["goalkeeper"].learning_rate == 0.01
    assert configs["goalkeeper"].batch_size == 128
    assert configs["outfield"].learning_rate == 0.0017320508075688787
    assert configs["outfield"].batch_size == 256
    for config in configs.values():
        assert config.random_seed == 42
        assert config.epochs == 50
        assert config.patience == 7
        assert config.device == "cpu"
        assert config.num_workers == 0
        assert (config.hidden_size1, config.hidden_size2) == (64, 32)


def test_training_duration_runs_each_population_once_with_final_test_evaluation():
    calls: list[tuple[ExperimentConfig, pd.DataFrame, bool]] = []
    population_data = {
        "goalkeeper": pd.DataFrame({"synthetic": [1]}),
        "outfield": pd.DataFrame({"synthetic": [2]}),
    }
    expected_results = {
        population: make_result(config)
        for population, config in make_milestone_6d_configs().items()
    }

    results = run_training_duration_experiments(
        population_data,
        runner_factory=lambda config: FakeRunner(
            config, calls, expected_results[config.population]
        ),
    )

    assert len(results) == 2
    assert len(calls) == 2
    assert [call[0].population for call in calls] == ["goalkeeper", "outfield"]
    assert all(call[2] is True for call in calls)
    assert results[0].experiment_result is expected_results["goalkeeper"]
    assert results[1].experiment_result is expected_results["outfield"]
    assert all(result.experiment_result.test_mae is not None for result in results)
    assert all(result.configuration.epochs == 50 for result in results)


def test_training_duration_requires_both_populations():
    with pytest.raises(ValueError, match="outfield"):
        run_training_duration_experiments(
            {"goalkeeper": pd.DataFrame({"synthetic": [1]})},
            runner_factory=lambda config: FakeRunner(config, [], make_result(config)),
        )