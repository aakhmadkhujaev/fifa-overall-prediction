# FIFA Overall Rating V1 Final Evaluation

## Evaluation Methodology

The official CPU V1 checkpoints were loaded without retraining. The raw dataset was loaded once, validated, and split into goalkeeper and outfield populations using the existing feature-engineering functions. For each population, the evaluator restored the checkpoint's fitted `Preprocessor` and model, reproduced the grouped deterministic split with `overall` as target, `player_id` as group, test size 0.15, validation size 0.15, and random state 42, then transformed only the held-out test features.

Inference was performed on CPU in batches of 2,048. Predictions, actual ratings, signed errors, absolute errors, directional counts/rates, fixed actual-rating bands, and largest absolute errors were calculated from the held-out test rows. No test value was used to select or modify a model configuration.

Dataset size: 161,583 rows. The goalkeeper population contains 17,970 rows and the outfield population contains 143,613 rows.

## Goalkeeper

Checkpoint: `models/fifa_overall_goalkeeper_v1.pt`

- Test rows: 2,835
- Features: 12
- MAE: 0.38301607966423035
- RMSE: 0.48369768261909485
- R2: 0.9957777825184166
- Mean error (prediction - actual): -0.029644231779444995
- Underprediction: 1,561 rows (0.5506172839506173)
- Overprediction: 1,274 rows (0.44938271604938274)
- Zero-error rows: 0
- Best epoch: 20
- Saved history epochs: 20
- Early stopping recorded in checkpoint: true

### Goalkeeper Rating Bands

| Rating range | Samples | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: | ---: |
| <60 | 734 | 0.3897337328835469 | 0.5028002216325743 | 0.014574056100455552 |
| 60-69 | 1,386 | 0.3772832972318751 | 0.46767052628812533 | -0.06262597858819782 |
| 70-79 | 627 | 0.3624216350451991 | 0.45772548079460046 | 0.006844321316319029 |
| 80-89 | 88 | 0.5640119205821644 | 0.7008184754956924 | -0.13898337971080432 |

Largest absolute errors are available in the structured JSON and prediction CSV; the largest observed goalkeeper absolute error was 3.2219276428222656.

## Outfield

Checkpoint: `models/fifa_overall_outfield_v1.pt`

- Test rows: 21,536
- Features: 35
- MAE: 0.7247869372367859
- RMSE: 0.9712936282157898
- R2: 0.9802714977413416
- Mean error (prediction - actual): -0.017078441992557068
- Underprediction: 11,760 rows (0.5460624071322436)
- Overprediction: 9,776 rows (0.4539375928677563)
- Zero-error rows: 0
- Best epoch: 27
- Saved history epochs: 27
- Early stopping recorded in checkpoint: false

### Outfield Rating Bands

| Rating range | Samples | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: | ---: |
| <60 | 3,721 | 0.8207478060641618 | 1.187507694195282 | 0.34244013486194275 |
| 60-69 | 11,402 | 0.7207641825720295 | 0.9527505642640941 | -0.01767673201361072 |
| 70-79 | 5,867 | 0.6626820867423693 | 0.8382246507086588 | -0.20430583443453917 |
| 80-89 | 543 | 0.81431127165343 | 1.0358577698013436 | -0.4329275619478735 |
| 90+ | 3 | 2.2429911295572915 | 2.247554473588642 | -2.2429911295572915 |

Largest absolute errors are available in the structured JSON and prediction CSV; the largest observed outfield absolute error was 9.836349487304688.

## Population Comparison

| Population | Test rows | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Goalkeeper | 2,835 | 0.38301607966423035 | 0.48369768261909485 | 0.9957777825184166 |
| Outfield | 21,536 | 0.7247869372367859 | 0.9712936282157898 | 0.9802714977413416 |

These are descriptive results for the two fixed populations and their fixed test distributions. They do not establish a causal explanation for the difference.

## Training History

The evaluator preserves the history saved inside each checkpoint, including training loss, validation loss, validation MAE/RMSE/R2, best epoch, best validation MAE, saved history length, and early-stopping state. The saved history ends at the checkpoint history captured by the existing trainer; it is not reconstructed or fabricated during evaluation.

## Artifacts

- `reports/generated/v1_final_evaluation.json`
- `reports/generated/v1_test_predictions.csv`
- `models/fifa_overall_goalkeeper_v1.pt`
- `models/fifa_overall_outfield_v1.pt`

The prediction CSV contains exactly `player_id`, `actual_overall`, `predicted_overall`, `error`, `absolute_error`, and `population`, with one row per held-out test sample.

## Validation Checks

- Both checkpoint files loaded through the V1 artifact loader.
- Both fitted preprocessors and models were reconstructed from checkpoint state.
- Checkpoint feature ordering matched the evaluation input ordering.
- Evaluated metrics matched the official V1 results within numerical tolerance.
- Prediction row counts matched the held-out split sizes: 2,835 goalkeeper and 21,536 outfield.
- Goalkeeper and outfield prediction player IDs were disjoint.
- Prediction and error values contained no NaN or infinite values.
- JSON and CSV artifacts were read back successfully.

## Limitations

This evaluation describes one deterministic grouped held-out split. Rating-band summaries with very small sample counts, especially the outfield 90+ band, should not be generalized beyond this test set. Error direction and largest-error records are descriptive and do not by themselves establish model bias, causation, or a required modeling change.
