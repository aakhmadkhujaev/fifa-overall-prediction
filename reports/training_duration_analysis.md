# Milestone 6D Training Duration Analysis

## Objective

Milestone 6D will test whether increasing the maximum training budget from 30 to 50 epochs provides additional validation improvement under the already optimized Milestone 6B learning rates and Milestone 6C batch sizes.

This is a controlled two-population experiment, not a hyperparameter search. The infrastructure runs one fixed configuration for goalkeepers and one for outfield players.

## Controlled Configurations

- Goalkeeper learning rate: `0.01`
- Goalkeeper batch size: `128`
- Outfield learning rate: `0.0017320508075688787`
- Outfield batch size: `256`
- Random seed: `42`
- Maximum epochs: `50`
- Patience: `7`
- Device: CPU
- Workers: `0`
- Architecture: `64 -> 32 -> 1`

Only the maximum epoch budget changes from Milestone 6C: `30 -> 50`. Optimizer, loss, preprocessing, feature definitions, grouped splitting, and evaluation metrics remain unchanged.

## Evaluation Protocol

The existing `ExperimentRunner` and `train_model()` implementation are reused. Validation MAE controls checkpoint selection and early stopping. The trainer restores the best validation state before the runner evaluates the test set. Each population configuration is run once with final test evaluation enabled; no test metric is used for model selection.

## Expected Comparison

The completed experiment will compare each population's best validation metrics and best epoch with its Milestone 6C result. A longer maximum budget will only be considered useful if the recorded validation evidence supports additional improvement. No real Milestone 6D training results have been generated yet.