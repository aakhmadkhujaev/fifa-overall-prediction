# Milestone 6C Batch-Size Search

Milestone 6C adds infrastructure for an exhaustive discrete batch-size search. It reuses `ExperimentConfig`, `ExperimentResult`, `RunnerProtocol`, and `ExperimentRunner`; it does not duplicate the training loop or change the trainer, model, preprocessing, feature definitions, or data splitting.

The configured candidates are `64`, `128`, `256`, `512`, and `1024`. Goalkeeper candidates use the Milestone 6B learning rate `0.01`; outfield candidates use `0.0017320508075688787`. Seed `42`, 30 epochs, patience `7`, CPU execution, zero workers, and the existing `64 -> 32 -> 1` architecture remain fixed.

Each candidate runs with `evaluate_test=False` and is ranked only by validation MAE. Candidate test metrics remain unavailable, and the selected candidate is evaluated on the test set exactly once with `evaluate_test=True`. The result records the candidate results, selected batch size, validation MAE, final result, executed trial count, and runtime.

## Goalkeeper Results

The fixed learning rate was `0.01`.

| Batch size | Validation MAE | Best epoch |
| ---: | ---: | ---: |
| `64` | `0.3853299916` | `13` |
| `128` | `0.3790021241` | `20` |
| `256` | `0.3913530111` | `27` |
| `512` | `0.4649064541` | `30` |
| `1024` | `0.7029228806` | `30` |

Selected batch size: `128`

- Selected validation MAE: `0.3790021241`
- Final test MAE: `0.383016`
- Final test RMSE: `0.483698`
- Final test R2: `0.995778`
- Candidate runs: `5`
- Total runtime: `43.269 s`
- Final evaluation runtime: `7.370 s`

Compared with the Milestone 6B goalkeeper result at batch size `256`, validation MAE decreased from `0.3913530111` to `0.3790021241`, and test MAE decreased from `0.395989` to `0.383016`. These results support an improvement for the tested batch-size choice, without establishing that batch size `128` is globally optimal.

## Outfield Results

The fixed learning rate was `0.0017320508075688787`.

| Batch size | Validation MAE | Best epoch |
| ---: | ---: | ---: |
| `64` | `0.7400665283` | `12` |
| `128` | `0.7289906144` | `19` |
| `256` | `0.7221816778` | `27` |
| `512` | `0.7549182773` | `28` |
| `1024` | `0.8821256161` | `30` |

Selected batch size: `256`

- Selected validation MAE: `0.7221816778`
- Final test MAE: `0.724787`
- Final test RMSE: `0.971294`
- Final test R2: `0.980271`
- Candidate runs: `5`
- Total runtime: `339.039 s`
- Final evaluation runtime: `60.021 s`

The selected outfield result matches the Milestone 6B result, which also used batch size `256`: validation MAE `0.7221816778`, test MAE `0.724787`, test RMSE `0.971294`, and test R2 `0.980271`. This batch-size search therefore did not improve the recorded outfield result.

## Limitations

- This was an exhaustive search over only the five tested batch sizes: `64`, `128`, `256`, `512`, and `1024`.
- Candidate selection used validation MAE only; candidate test loaders were not evaluated.
- Exactly one final test evaluation was performed for each selected population configuration.
- The experiment optimized batch size only. Learning rate, seed, epochs, patience, device, architecture, optimizer, loss, preprocessing, features, and grouped data splits remained fixed.
- The selected batch size is the best among these tested candidates, not a proof of global optimality.

## Artifact

The structured results and audit fields are stored in:

`reports/generated/milestone_6c_batch_size_search.json`