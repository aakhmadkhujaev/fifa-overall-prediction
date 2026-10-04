"""Two-stage, validation-driven learning-rate search for FIFA experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Tuple

import pandas as pd

from src.experiments.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    Population,
    RunnerProtocol,
)


RunnerFactory = Callable[[ExperimentConfig], RunnerProtocol]


def make_milestone_6b_search_config(
    search_name: str,
    population: Population,
) -> "LearningRateSearchConfig":
    """Create the approved Milestone 6B search configuration."""
    return LearningRateSearchConfig(
        search_name=search_name,
        min_learning_rate=1e-4,
        max_learning_rate=1e-2,
        coarse_trials=7,
        fine_trials=5,
        refinement_factor=3.0,
        fixed_config=ExperimentConfig(
            experiment_name=f"{search_name}-fixed",
            population=population,
        ),
    )


@dataclass(frozen=True)
class LearningRateSearchConfig:
    """Validated settings for one population-specific learning-rate search."""

    search_name: str
    min_learning_rate: float
    max_learning_rate: float
    coarse_trials: int
    fine_trials: int
    fixed_config: ExperimentConfig
    refinement_factor: float = 3.0

    def __post_init__(self) -> None:
        if not self.search_name.strip():
            raise ValueError("search_name must not be empty.")
        if (
            not math.isfinite(self.min_learning_rate)
            or self.min_learning_rate <= 0
        ):
            raise ValueError("min_learning_rate must be finite and greater than zero.")
        if (
            not math.isfinite(self.max_learning_rate)
            or self.max_learning_rate <= self.min_learning_rate
        ):
            raise ValueError(
                "max_learning_rate must be finite and greater than min_learning_rate."
            )
        if (
            isinstance(self.coarse_trials, bool)
            or not isinstance(self.coarse_trials, int)
            or self.coarse_trials < 2
        ):
            raise ValueError("coarse_trials must be at least 2.")
        if (
            isinstance(self.fine_trials, bool)
            or not isinstance(self.fine_trials, int)
            or self.fine_trials < 2
        ):
            raise ValueError("fine_trials must be at least 2.")
        if (
            not math.isfinite(self.refinement_factor)
            or self.refinement_factor <= 1
        ):
            raise ValueError("refinement_factor must be finite and greater than 1.")
        _validate_fixed_config(self.fixed_config)


@dataclass(frozen=True)
class LearningRateSearchResult:
    """Immutable record of all candidates and the validation-selected winner."""

    search_name: str
    population: Population
    coarse_results: Tuple[ExperimentResult, ...]
    fine_results: Tuple[ExperimentResult, ...]
    best_learning_rate: float
    best_validation_mae: float
    best_experiment_result: ExperimentResult
    total_trials: int
    total_duration_seconds: float


def generate_coarse_learning_rates(
    config: LearningRateSearchConfig,
) -> Tuple[float, ...]:
    """Generate deterministic logarithmically spaced coarse candidates."""
    return _logarithmic_space(
        config.min_learning_rate,
        config.max_learning_rate,
        config.coarse_trials,
    )


def generate_fine_learning_rates(
    config: LearningRateSearchConfig,
    coarse_winner: float,
) -> Tuple[float, ...]:
    """Generate a clipped, denser logarithmic region around a coarse winner."""
    if not math.isfinite(coarse_winner) or coarse_winner <= 0:
        raise ValueError("coarse_winner must be finite and greater than zero.")
    if not config.min_learning_rate <= coarse_winner <= config.max_learning_rate:
        raise ValueError("coarse_winner must be inside the configured search range.")

    lower_bound = max(
        config.min_learning_rate,
        coarse_winner / config.refinement_factor,
    )
    upper_bound = min(
        config.max_learning_rate,
        coarse_winner * config.refinement_factor,
    )
    return _logarithmic_space(lower_bound, upper_bound, config.fine_trials)


class LearningRateSearcher:
    """Run two-stage learning-rate candidates through the existing runner."""

    def __init__(
        self,
        config: LearningRateSearchConfig,
        *,
        runner_factory: RunnerFactory = ExperimentRunner,
    ) -> None:
        self.config = config
        self._runner_factory = runner_factory

    def run(self, population_data: pd.DataFrame) -> LearningRateSearchResult:
        """Search learning rates and select the best tested validation MAE."""
        started = time.perf_counter()
        coarse_results = self._run_candidates(
            population_data,
            generate_coarse_learning_rates(self.config),
            stage="coarse",
        )
        coarse_winner = min(coarse_results, key=lambda result: result.validation_mae)
        fine_rates = generate_fine_learning_rates(
            self.config,
            coarse_winner.configuration.learning_rate,
        )
        coarse_rates = {
            result.configuration.learning_rate for result in coarse_results
        }
        fine_rates = _deduplicate_learning_rates(fine_rates, coarse_rates)
        fine_results = self._run_candidates(
            population_data,
            fine_rates,
            stage="fine",
        )
        all_results = coarse_results + fine_results
        selected_result = min(all_results, key=lambda result: result.validation_mae)
        final_result = self._runner_factory(selected_result.configuration).run(
            population_data,
            evaluate_test=True,
        )

        return LearningRateSearchResult(
            search_name=self.config.search_name,
            population=self.config.fixed_config.population,
            coarse_results=coarse_results,
            fine_results=fine_results,
            best_learning_rate=selected_result.configuration.learning_rate,
            best_validation_mae=selected_result.validation_mae,
            best_experiment_result=final_result,
            total_trials=len(all_results),
            total_duration_seconds=time.perf_counter() - started,
        )

    def _run_candidates(
        self,
        population_data: pd.DataFrame,
        learning_rates: Tuple[float, ...],
        *,
        stage: str,
    ) -> Tuple[ExperimentResult, ...]:
        results = []
        for index, learning_rate in enumerate(learning_rates, start=1):
            candidate_config = replace(
                self.config.fixed_config,
                experiment_name=(
                    f"{self.config.search_name}-{stage}-{index}"
                ),
                learning_rate=learning_rate,
            )
            runner = self._runner_factory(candidate_config)
            results.append(runner.run(population_data, evaluate_test=False))
        return tuple(results)


def _logarithmic_space(
    minimum: float,
    maximum: float,
    count: int,
) -> Tuple[float, ...]:
    if count < 2:
        raise ValueError("Logarithmic search requires at least two candidates.")
    log_minimum = math.log(minimum)
    step = (math.log(maximum) - log_minimum) / (count - 1)
    candidates = [math.exp(log_minimum + index * step) for index in range(count)]
    candidates[0] = minimum
    candidates[-1] = maximum
    return tuple(candidates)


def _deduplicate_learning_rates(
    candidates: Tuple[float, ...],
    already_evaluated: set[float],
) -> Tuple[float, ...]:
    """Keep candidate order while removing rates already evaluated."""
    unique_candidates = []
    seen = set(already_evaluated)
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return tuple(unique_candidates)


def _validate_fixed_config(config: ExperimentConfig) -> None:
    expected_values = {
        "random_seed": 42,
        "batch_size": 256,
        "epochs": 30,
        "patience": 7,
        "device": "cpu",
        "num_workers": 0,
        "hidden_size1": 64,
        "hidden_size2": 32,
    }
    mismatches = {
        field_name: (getattr(config, field_name), expected)
        for field_name, expected in expected_values.items()
        if getattr(config, field_name) != expected
    }
    if mismatches:
        raise ValueError(
            "fixed_config must preserve Milestone 6B training settings: "
            f"{mismatches}"
        )