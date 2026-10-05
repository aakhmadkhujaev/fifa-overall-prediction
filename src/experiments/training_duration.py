"""Controlled Milestone 6D training-duration experiment orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

import pandas as pd

from src.experiments.experiment import (
    ExperimentConfig,
    ExperimentResult,
    Population,
    RunnerProtocol,
)


RunnerFactory = Callable[[ExperimentConfig], RunnerProtocol]
OPTIMIZED_LEARNING_RATES = {
    "goalkeeper": 0.01,
    "outfield": 0.0017320508075688787,
}
OPTIMIZED_BATCH_SIZES = {"goalkeeper": 128, "outfield": 256}


@dataclass(frozen=True)
class TrainingDurationResult:
    """Structured result for one fixed-population duration experiment."""

    population: Population
    configuration: ExperimentConfig
    experiment_result: ExperimentResult
    runtime_seconds: float


def make_milestone_6d_configs() -> dict[Population, ExperimentConfig]:
    """Build the fixed 50-epoch configurations for both populations."""
    return {
        population: ExperimentConfig(
            experiment_name=f"milestone-6d-{population}",
            population=population,
            random_seed=42,
            batch_size=OPTIMIZED_BATCH_SIZES[population],
            learning_rate=OPTIMIZED_LEARNING_RATES[population],
            epochs=50,
            patience=7,
            device="cpu",
            num_workers=0,
            hidden_size1=64,
            hidden_size2=32,
        )
        for population in ("goalkeeper", "outfield")
    }


def run_training_duration_experiments(
    population_data: Mapping[Population, pd.DataFrame],
    *,
    runner_factory: RunnerFactory,
) -> Tuple[TrainingDurationResult, ...]:
    """Run each fixed population configuration exactly once."""
    results = []
    for population, configuration in make_milestone_6d_configs().items():
        if population not in population_data:
            raise ValueError(f"Missing data for population: {population}.")
        started = time.perf_counter()
        experiment_result = runner_factory(configuration).run(
            population_data[population],
            evaluate_test=True,
        )
        results.append(
            TrainingDurationResult(
                population=population,
                configuration=configuration,
                experiment_result=experiment_result,
                runtime_seconds=time.perf_counter() - started,
            )
        )
    return tuple(results)