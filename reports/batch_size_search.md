# Milestone 6C Batch-Size Search

Milestone 6C adds infrastructure for an exhaustive discrete batch-size search. It reuses `ExperimentConfig`, `ExperimentResult`, `RunnerProtocol`, and `ExperimentRunner`; it does not duplicate the training loop or change the trainer, model, preprocessing, feature definitions, or data splitting.

The configured candidates are `64`, `128`, `256`, `512`, and `1024`. Goalkeeper candidates use the Milestone 6B learning rate `0.01`; outfield candidates use `0.0017320508075688787`. Seed `42`, 30 epochs, patience `7`, CPU execution, zero workers, and the existing `64 -> 32 -> 1` architecture remain fixed.

Each candidate runs with `evaluate_test=False` and is ranked only by validation MAE. Candidate test metrics remain unavailable, and the selected candidate is evaluated on the test set exactly once with `evaluate_test=True`. The result records the candidate results, selected batch size, validation MAE, final result, executed trial count, and runtime.

This is infrastructure only. No real FIFA batch-size search has been run yet, and this report contains no experiment results.