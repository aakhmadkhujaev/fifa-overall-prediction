"""Exhaustive, validation-driven batch-size search for FIFA experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Tuple

import pandas as pd

from src.experiments.experiment import (
    ExperimentConfig,
    ExperimentResult,
    Population,
    RunnerProtocol,
)


BatchSize = int
RunnerFactory = Callable[[ExperimentConfig], RunnerProtocol]
DEFAULT_BATCH_SIZES: Tuple[BatchSize, ...] = (64, 128, 256, 512, 1024)
FIXED_LEARNING_RATES = {
    "goalkeeper": 0.01,
    "outfield": 0.0017320508075688787,
}


def make_milestone_6c_search_config(
    search_name: str,
    population: Population,
) -> "BatchSizeSearchConfig":
    """Create the fixed Milestone 6C configuration for one population."""
    return BatchSizeSearchConfig(
        search_name=search_name,
        population=population,
        fixed_config=ExperimentConfig(
            experiment_name=f"{search_name}-fixed",
            population=population,
            learning_rate=FIXED_LEARNING_RATES[population],
        ),
    )


@dataclass(frozen=True)
class BatchSizeSearchConfig:
    """Validated settings for one exhaustive population-specific search."""

    search_name: str
    population: Population
    fixed_config: ExperimentConfig
    batch_sizes: Tuple[BatchSize, ...] = DEFAULT_BATCH_SIZES

    def __post_init__(self) -> None:
        if not self.search_name.strip():
            raise ValueError("search_name must not be empty.")
        if self.population not in {"goalkeeper", "outfield"}:
            raise ValueError("population must be 'goalkeeper' or 'outfield'.")
        if self.fixed_config.population != self.population:
            raise ValueError("fixed_config population must match population.")
        expected_learning_rate = FIXED_LEARNING_RATES[self.population]
        if not math.isclose(
            self.fixed_config.learning_rate,
            expected_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "fixed_config must use the Milestone 6B learning rate for "
                f"{self.population}."
            )
        _validate_fixed_config(self.fixed_config)
        if not self.batch_sizes:
            raise ValueError("batch_sizes must contain at least one value.")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates.")
        for batch_size in self.batch_sizes:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise ValueError("batch_sizes must contain only integers.")
            if batch_size <= 0:
                raise ValueError("batch_sizes must contain only positive integers.")


@dataclass(frozen=True)
class BatchSizeSearchResult:
    """Immutable record of candidate results and the selected final result."""

    search_name: str
    population: Population
    candidate_results: Tuple[ExperimentResult, ...]
    selected_batch_size: BatchSize
    selected_validation_mae: float
    best_experiment_result: ExperimentResult
    total_trials: int
    total_runtime_seconds: float


def generate_batch_size_candidates(
    config: BatchSizeSearchConfig,
) -> Tuple[BatchSize, ...]:
    """Return the configured exhaustive batch-size candidates in order."""
    return tuple(config.batch_sizes)


class BatchSizeSearcher:
    """Run exhaustive batch-size candidates through the existing runner."""

    def __init__(
        self,
        config: BatchSizeSearchConfig,
        *,
        runner_factory: RunnerFactory,
    ) -> None:
        self.config = config
        self._runner_factory = runner_factory

    def run(self, population_data: pd.DataFrame) -> BatchSizeSearchResult:
        """Select by validation MAE, then evaluate the winner on test once."""
        started = time.perf_counter()
        candidate_results = tuple(
            self._run_candidate(population_data, batch_size)
            for batch_size in generate_batch_size_candidates(self.config)
        )
        selected_result = min(
            candidate_results,
            key=lambda result: result.validation_mae,
        )
        final_result = self._runner_factory(
            selected_result.configuration
        ).run(population_data, evaluate_test=True)

        return BatchSizeSearchResult(
            search_name=self.config.search_name,
            population=self.config.population,
            candidate_results=candidate_results,
            selected_batch_size=selected_result.configuration.batch_size,
            selected_validation_mae=selected_result.validation_mae,
            best_experiment_result=final_result,
            total_trials=len(candidate_results),
            total_runtime_seconds=time.perf_counter() - started,
        )

    def _run_candidate(
        self,
        population_data: pd.DataFrame,
        batch_size: BatchSize,
    ) -> ExperimentResult:
        candidate_config = replace(
            self.config.fixed_config,
            experiment_name=f"{self.config.search_name}-batch-{batch_size}",
            batch_size=batch_size,
        )
        return self._runner_factory(candidate_config).run(
            population_data,
            evaluate_test=False,
        )


def _validate_fixed_config(config: ExperimentConfig) -> None:
    expected_values = {
        "random_seed": 42,
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
            "fixed_config must preserve Milestone 6C training settings: "
            f"{mismatches}"
        )