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

## Actual Milestone 6D Results

The two fixed configurations were each run once on the full dataset. The recorded results are also stored in `reports/generated/milestone_6d_training_duration.json`.

| Population | Best epoch | Best validation MAE | Test MAE | Test RMSE | Test R2 | Runtime (seconds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Goalkeeper | 20 | 0.379002 | 0.383016 | 0.483698 | 0.995778 | 11.0358 |
| Outfield | 27 | 0.722182 | 0.724787 | 0.971294 | 0.980271 | 57.9215 |

### Goalkeeper

The best epoch remained 20, so it did not move beyond the previous 30-epoch limit. The best validation MAE was `0.379002`, unchanged from the Milestone 6C selected run. Test MAE was also unchanged at `0.383016` compared with the 6C value of `0.383016`. Early stopping occurred after seven non-improving epochs, at epoch 27.

For the final V1 configuration, the 50-epoch maximum did not provide additional measured benefit for goalkeepers. The optimized learning rate `0.01`, batch size `128`, and validation-MAE checkpoint selection remain supported; a 30-epoch maximum would have contained the selected checkpoint in this run.

### Outfield

The best epoch remained 27, so it did not move beyond the previous 30-epoch limit. The best validation MAE was `0.722182`, unchanged from the Milestone 6C selected run. Test MAE was unchanged at `0.724787` compared with the 6C value of `0.724787`. Early stopping occurred after seven non-improving epochs, at epoch 34.

For the final V1 configuration, the 50-epoch maximum did not provide additional measured benefit for outfield players. The optimized learning rate `0.0017320508075688787`, batch size `256`, and validation-MAE checkpoint selection remain supported; a 30-epoch maximum would have contained the selected checkpoint in this run.

Overall, these results support retaining the Milestone 6C learning rates and batch sizes with validation-based checkpoint restoration and early stopping. They do not establish that 50 epochs is globally optimal.