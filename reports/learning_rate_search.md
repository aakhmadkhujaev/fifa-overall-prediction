# Milestone 6B Learning-Rate Search

## Experiment Objective

Milestone 6B optimized the learning rate using validation MAE as the selection metric. Goalkeeper and outfield players were searched as separate populations using the existing `LearningRateSearcher` and `ExperimentRunner` infrastructure.

## Search Configuration

The search covered learning rates from `1e-4` through `1e-2` with 7 logarithmically spaced coarse trials and 5 fine trials. The fine-search region used a refinement factor of `3.0` around the coarse winner.

All candidate experiments used the fixed configuration below:

- Random seed: `42`
- Batch size: `256`
- Epochs: `30`
- Patience: `7`
- Device: CPU
- Model architecture: `64 -> 32 -> 1`
- Candidate deduplication: fine candidates already evaluated during the coarse stage were skipped

## Experimental Protocol

- Every coarse and fine candidate ran with `evaluate_test=False`.
- Candidates were selected using validation MAE only.
- Candidate test loaders were not evaluated.
- Exactly one final test evaluation was performed for the selected configuration in each population.
- Test MAE, RMSE, and R2 were not used for learning-rate selection.
- The selected learning rate is the best tested candidate, not a proven global optimum.

## Results: Goalkeepers

### Candidate Results

| Stage | Learning rate | Validation MAE |
| --- | ---: | ---: |
| Coarse | `0.0001` | `24.7770195007` |
| Coarse | `0.0002154434690031884` | `11.9101552963` |
| Coarse | `0.0004641588833612784` | `3.8483734131` |
| Coarse | `0.0010000000000000002` | `0.9253343940` |
| Coarse | `0.0021544346900318864` | `0.6510482430` |
| Coarse | `0.004641588833612781` | `0.4440564811` |
| Coarse | `0.01` | `0.3913530111` |
| Fine | `0.0033333333333333335` | `0.5076324344` |
| Fine | `0.00438691337650831` | `0.4536960423` |
| Fine | `0.005773502691896262` | `0.4214640260` |
| Fine | `0.0075983568565159264` | `0.4032679200` |

The raw fine grid also contained `0.01`, which was skipped because it had already been evaluated during the coarse stage.

### Selected Result

| Selected learning rate | Selected validation MAE | Final test MAE | Final test RMSE | Final test R2 | Best epoch | Candidate runs | Total runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | `0.3913530111` | `0.395989` | `0.516361` | `0.995188` | `27` | `11` | `79.644 s` |

The goalkeeper selection was `0.01`, the upper boundary of the tested range. It continued to improve relative to the lower tested rates, but this does not establish that `0.01` is globally optimal. A future experiment could investigate higher learning rates.

## Results: Outfield Players

### Candidate Results

| Stage | Learning rate | Validation MAE |
| --- | ---: | ---: |
| Coarse | `0.0001` | `1.2032822371` |
| Coarse | `0.0002154434690031884` | `0.8365021944` |
| Coarse | `0.0004641588833612784` | `0.7522351146` |
| Coarse | `0.0010000000000000002` | `0.7280161977` |
| Coarse | `0.0021544346900318864` | `0.7396999002` |
| Coarse | `0.004641588833612781` | `0.7412030697` |
| Coarse | `0.01` | `0.7685672641` |
| Fine | `0.00033333333333333343` | `0.7775193453` |
| Fine | `0.000577350269189626` | `0.7470659018` |
| Fine | `0.0017320508075688787` | `0.7221816778` |
| Fine | `0.003000000000000001` | `0.7321598530` |

The raw fine grid also contained `0.0010000000000000002`, which was skipped because it had already been evaluated during the coarse stage.

### Selected Result

| Selected learning rate | Selected validation MAE | Final test MAE | Final test RMSE | Final test R2 | Best epoch | Candidate runs | Total runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0017320508075688787` | `0.7221816778` | `0.724787` | `0.971294` | `0.980271` | `27` | `11` | `583.338 s` |

The selected outfield learning rate produced a modest improvement over the recorded `0.0010000000000000002` baseline candidate, whose validation MAE was `0.7280161977`, reaching `0.7221816778`.

## Limitations

- The search covered only `1e-4` through `1e-2`.
- The selected learning rate is the best among the tested candidates and is not proven globally optimal.
- This experiment optimized learning rate only.
- Batch size, epoch count, patience, device, random seed, model architecture, preprocessing, feature selection, and data splitting remained fixed.
- No claims are made about performance outside the tested candidate grids.

## Artifact

The complete structured experiment output, including configurations, all candidate results, final metrics, runtimes, and protocol audit fields, is stored in:

`reports/generated/milestone_6b_learning_rate_search.json`