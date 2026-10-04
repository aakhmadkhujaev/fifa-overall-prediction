from dataclasses import replace
from math import isfinite
from typing import Optional

import pandas as pd
import pytest

from src.experiments.experiment import ExperimentConfig, ExperimentResult, Population
from src.experiments.learning_rate_search import (
    LearningRateSearchConfig,
    LearningRateSearcher,
    generate_coarse_learning_rates,
    generate_fine_learning_rates,
    make_milestone_6b_search_config,
)


def make_search_config(population: Population = "outfield"):
    return LearningRateSearchConfig(
        search_name="smoke-search",
        min_learning_rate=1e-4,
        max_learning_rate=5e-3,
        coarse_trials=4,
        fine_trials=3,
        fixed_config=ExperimentConfig("fixed", population),
        refinement_factor=2.0,
    )


@pytest.mark.parametrize("population", ["goalkeeper", "outfield"])
def test_milestone_6b_config_uses_approved_fixed_settings(population):
    config = make_milestone_6b_search_config("approved-search", population)

    assert (config.min_learning_rate, config.max_learning_rate) == (1e-4, 1e-2)
    assert (config.coarse_trials, config.fine_trials) == (7, 5)
    assert config.refinement_factor == 3.0
    assert config.fixed_config.population == population
    assert config.fixed_config.random_seed == 42
    assert config.fixed_config.batch_size == 256
    assert config.fixed_config.epochs == 30
    assert config.fixed_config.patience == 7
    assert config.fixed_config.device == "cpu"
    assert config.fixed_config.num_workers == 0
    assert (config.fixed_config.hidden_size1, config.fixed_config.hidden_size2) == (64, 32)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"search_name": ""}, "search_name"),
        ({"min_learning_rate": 0}, "min_learning_rate"),
        ({"min_learning_rate": float("inf")}, "min_learning_rate"),
        ({"max_learning_rate": 1e-4}, "max_learning_rate"),
        ({"coarse_trials": 1}, "coarse_trials"),
        ({"fine_trials": 1}, "fine_trials"),
        ({"refinement_factor": 1}, "refinement_factor"),
    ],
)
def test_search_configuration_rejects_invalid_values(changes, message):
    with pytest.raises(ValueError, match=message):
        LearningRateSearchConfig(**{
            "search_name": changes.get("search_name", "smoke-search"),
            "min_learning_rate": changes.get("min_learning_rate", 1e-4),
            "max_learning_rate": changes.get("max_learning_rate", 5e-3),
            "coarse_trials": changes.get("coarse_trials", 4),
            "fine_trials": changes.get("fine_trials", 3),
            "fixed_config": ExperimentConfig("fixed", "outfield"),
            "refinement_factor": changes.get("refinement_factor", 2.0),
        })

    with pytest.raises(ValueError, match="Milestone 6B"):
        LearningRateSearchConfig(
            "invalid-fixed",
            1e-4,
            5e-3,
            4,
            3,
            replace(ExperimentConfig("fixed", "outfield"), batch_size=32),
        )


def test_coarse_candidates_are_deterministic_logarithmic_and_in_range():
    config = make_search_config()

    first = generate_coarse_learning_rates(config)
    second = generate_coarse_learning_rates(config)
    ratios = [first[index + 1] / first[index] for index in range(len(first) - 1)]

    assert first == second
    assert len(first) == config.coarse_trials
    assert first[0] == pytest.approx(config.min_learning_rate)
    assert first[-1] == pytest.approx(config.max_learning_rate)
    assert all(config.min_learning_rate <= rate <= config.max_learning_rate for rate in first)
    assert ratios[1:] == pytest.approx(ratios[:-1])


def test_fine_candidates_are_deterministic_and_centered_on_coarse_winner():
    config = make_search_config()
    winner = 1e-3

    candidates = generate_fine_learning_rates(config, winner)

    assert candidates == generate_fine_learning_rates(config, winner)
    assert len(candidates) == config.fine_trials
    assert candidates[0] == pytest.approx(winner / config.refinement_factor)
    assert candidates[-1] == pytest.approx(winner * config.refinement_factor)
    assert all(isfinite(rate) and rate > 0 for rate in candidates)


def make_result(
    config: ExperimentConfig,
    validation_mae: float,
    test_mae: Optional[float],
):
    return ExperimentResult(
        experiment_name=config.experiment_name,
        population=config.population,
        configuration=config,
        best_epoch=1,
        best_validation_mae=validation_mae,
        validation_mae=validation_mae,
        validation_rmse=validation_mae,
        validation_r2=0.5,
        test_mae=test_mae,
        test_rmse=test_mae,
        test_r2=0.5,
        training_duration_seconds=0.01,
    )


class FakeRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        calls: list[tuple[ExperimentConfig, pd.DataFrame, bool]],
        metrics: dict[float, tuple[float, float]],
    ) -> None:
        self.config = config
        self.calls = calls
        self.metrics = metrics

    def run(
        self,
        population_data: pd.DataFrame,
        *,
        evaluate_test: bool = True,
    ) -> ExperimentResult:
        self.calls.append((self.config, population_data, evaluate_test))
        validation_mae, test_mae = self.metrics.get(
            self.config.learning_rate, (1.0, 1.0)
        )
        return make_result(
            self.config,
            validation_mae,
            test_mae if evaluate_test else None,
        )


@pytest.mark.parametrize("population", ["goalkeeper", "outfield"])
def test_search_executes_candidates_with_only_learning_rate_changed(population):
    config = make_search_config(population)
    calls = []
    metrics = {}
    for rate in generate_coarse_learning_rates(config):
        metrics[rate] = (1.0, 1.0)
    winner = generate_coarse_learning_rates(config)[1]
    metrics[winner] = (0.5, 1.0)
    for rate in generate_fine_learning_rates(config, winner):
        metrics[rate] = (1.0, 1.0)

    def runner_factory(candidate_config):
        return FakeRunner(candidate_config, calls, metrics)

    result = LearningRateSearcher(config, runner_factory=runner_factory).run(
        pd.DataFrame({"synthetic": [1]})
    )

    candidate_calls = [call for call in calls if not call[2]]
    assert len({candidate_config.learning_rate for candidate_config, _, _ in candidate_calls}) == len(candidate_calls)
    assert result.population == population
    for candidate_config, _, evaluate_test in candidate_calls:
        assert evaluate_test is False
        assert candidate_config.population == population
        assert candidate_config.random_seed == 42
        assert candidate_config.batch_size == 256
        assert candidate_config.epochs == 30
        assert candidate_config.patience == 7
        assert candidate_config.device == "cpu"
        assert candidate_config.num_workers == 0
        assert (candidate_config.hidden_size1, candidate_config.hidden_size2) == (64, 32)


def test_search_selects_by_validation_mae_not_test_mae():
    config = make_search_config()
    calls = []
    metrics = {}
    coarse_rates = generate_coarse_learning_rates(config)
    fine_rates = generate_fine_learning_rates(config, coarse_rates[1])
    for rate in fine_rates:
        metrics[rate] = (0.80, 0.10)
    for rate in coarse_rates:
        metrics[rate] = (0.75, 0.50)
    metrics[coarse_rates[1]] = (0.70, 0.90)

    result = LearningRateSearcher(
        config,
        runner_factory=lambda candidate_config: FakeRunner(
            candidate_config, calls, metrics
        ),
    ).run(pd.DataFrame({"synthetic": [1]}))

    assert result.best_learning_rate == coarse_rates[1]
    assert result.best_validation_mae == 0.70
    assert result.best_experiment_result.test_mae == 0.90
    assert result.best_experiment_result.configuration.learning_rate == coarse_rates[1]
    assert result.total_trials == len(result.coarse_results) + len(result.fine_results)
    assert len(result.coarse_results) == config.coarse_trials
    assert len(result.fine_results) <= config.fine_trials


def test_search_does_not_execute_duplicate_coarse_and_fine_rates():
    config = make_search_config()
    coarse_rates = generate_coarse_learning_rates(config)
    coarse_winner = coarse_rates[1]
    fine_rates = generate_fine_learning_rates(config, coarse_winner)
    expected_rates = list(coarse_rates)
    expected_rates.extend(
        rate for rate in fine_rates if rate not in expected_rates
    )
    calls = []
    metrics = {rate: (1.0, 1.0) for rate in expected_rates}
    metrics[coarse_winner] = (0.1, 1.0)

    result = LearningRateSearcher(
        config,
        runner_factory=lambda candidate_config: FakeRunner(
            candidate_config, calls, metrics
        ),
    ).run(pd.DataFrame({"synthetic": [1]}))

    executed_rates = [candidate_config.learning_rate for candidate_config, _, evaluate_test in calls if not evaluate_test]
    assert executed_rates == expected_rates
    assert len(executed_rates) == len(set(executed_rates))
    assert result.total_trials == len(expected_rates)
    assert len(result.fine_results) < config.fine_trials
    assert sum(evaluate_test for _, _, evaluate_test in calls) == 1
    assert calls[-1][0].learning_rate == result.best_learning_rate
    assert calls[-1][2] is True
    assert all(candidate_result.test_mae is None for candidate_result in result.coarse_results)
    assert all(candidate_result.test_mae is None for candidate_result in result.fine_results)