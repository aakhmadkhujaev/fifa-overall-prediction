# Baseline Error Analysis

## Objective

This analysis characterizes the behavior and limitations of the existing 5C goalkeeper and outfield baseline models. It does not redesign, tune, retrain, or compare the populations as a ranking.

## Data/model sources

- Source data: the repository's raw male-player FIFA dataset.
- Features: the existing `select_features` definitions (12 goalkeeper features and 35 outfield features).
- Models: the existing CPU PyTorch checkpoints and `FIFAOverallModel` architecture.
- Split/preprocessing: the existing `Preprocessor.process` grouped split with `random_state=42`; imputation and scaling were fit on training rows and only transformed the validation and test rows.

## Evaluation methodology

Predictions were generated for every row in each complete test split. Residual means use `prediction - actual`; absolute error is `abs(residual)`. MAE, RMSE, R2, residual direction, fixed rating bands, large-error rows, and Spearman feature/error associations were computed from those predictions. No test values were used to fit preprocessing.

## Population comparison

This table is descriptive only. Differences reflect the two populations, their feature sets, and their test distributions; they do not establish that one model is preferable.

| population | test_samples | mae | rmse | r2 | mean_error | underprediction_rate | overprediction_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Goalkeeper | 2835 | 0.994456 | 1.393652 | 0.964949 | 0.011259 | 0.577072 | 0.422928 |
| Outfield | 21536 | 0.734863 | 0.987386 | 0.979612 | 0.065663 | 0.505433 | 0.494567 |

## Goalkeeper analysis

Checkpoint: `fifa_overall_goalkeeper_baseline.pt`. The checkpoint was loaded on CPU and evaluated on 2,835 test rows after the existing grouped split and train-fitted preprocessing.

### Test metrics

| Metric | Value |
| --- | ---: |
| MAE | 0.994456 |
| RMSE | 1.393652 |
| R2 | 0.964949 |
| Mean error (prediction - actual) | 0.011259 |
| Mean absolute error | 0.994456 |
| Underprediction rate | 57.71% |
| Overprediction rate | 42.29% |

### Rating ranges

Bins are fixed FIFA-style bands selected before looking at error values. Empty bands are omitted; counts therefore describe the actual test distribution.

| rating_range | samples | actual_mean | mean_error | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| <60 | 734 | 55.1526 | 0.1874 | 1.0994 | 1.5524 |
| 60-69 | 1386 | 64.5094 | -0.0888 | 1.0014 | 1.3817 |
| 70-79 | 627 | 72.7751 | 0.0356 | 0.8233 | 1.1613 |
| 80-89 | 88 | 82.4091 | -0.0550 | 1.2290 | 1.6707 |

### Residuals and large errors

Residuals are defined as prediction minus actual. A positive mean error indicates overall overprediction; a negative value indicates overall underprediction. The largest absolute errors are shown without player names or IDs:

| actual | prediction | residual | absolute_error | age | height_cm | weight_kg |
| --- | --- | --- | --- | --- | --- | --- |
| 64.000 | 74.867 | 10.867 | 10.867 | 18.000 | 202.000 | 85.000 |
| 70.000 | 80.052 | 10.052 | 10.052 | 33.000 | 197.000 | 98.000 |
| 67.000 | 76.363 | 9.363 | 9.363 | 19.000 | 202.000 | 85.000 |
| 67.000 | 76.176 | 9.176 | 9.176 | 22.000 | 202.000 | 85.000 |
| 67.000 | 76.115 | 9.115 | 9.115 | 20.000 | 202.000 | 85.000 |
| 49.000 | 57.949 | 8.949 | 8.949 | 42.000 | 188.000 | 86.000 |
| 48.000 | 56.822 | 8.822 | 8.822 | 44.000 | 188.000 | 86.000 |
| 55.000 | 63.480 | 8.480 | 8.480 | 37.000 | 187.000 | 79.000 |
| 69.000 | 77.020 | 8.020 | 8.020 | 21.000 | 202.000 | 85.000 |
| 74.000 | 81.582 | 7.582 | 7.582 | 32.000 | 197.000 | 98.000 |

![Goalkeeper actual versus predicted](baseline_error_analysis_figures/goalkeeper_actual_vs_predicted.png)
![Goalkeeper residual distribution](baseline_error_analysis_figures/goalkeeper_residual_distribution.png)
![Goalkeeper error by rating range](baseline_error_analysis_figures/goalkeeper_error_by_rating_range.png)

### Feature/error observations

The five largest absolute Spearman associations with absolute error are shown below. These are observational associations within this test set, not causal effects or evidence that a feature should be added, removed, or transformed.

| feature | signed_error_spearman | absolute_error_spearman |
| --- | --- | --- |
| goalkeeping_reflexes | 0.0097 | -0.1186 |
| goalkeeping_diving | 0.0033 | -0.1139 |
| goalkeeping_kicking | 0.0007 | -0.0852 |
| goalkeeping_handling | 0.0232 | -0.0850 |
| goalkeeping_positioning | 0.0321 | -0.0791 |


## Outfield analysis

Checkpoint: `fifa_overall_outfield_baseline.pt`. The checkpoint was loaded on CPU and evaluated on 21,536 test rows after the existing grouped split and train-fitted preprocessing.

### Test metrics

| Metric | Value |
| --- | ---: |
| MAE | 0.734863 |
| RMSE | 0.987386 |
| R2 | 0.979612 |
| Mean error (prediction - actual) | 0.065663 |
| Mean absolute error | 0.734863 |
| Underprediction rate | 50.54% |
| Overprediction rate | 49.46% |

### Rating ranges

Bins are fixed FIFA-style bands selected before looking at error values. Empty bands are omitted; counts therefore describe the actual test distribution.

| rating_range | samples | actual_mean | mean_error | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| <60 | 3721 | 55.5020 | 0.3520 | 0.8350 | 1.2018 |
| 60-69 | 11402 | 64.7987 | 0.0594 | 0.7334 | 0.9720 |
| 70-79 | 5867 | 73.1565 | -0.0778 | 0.6704 | 0.8568 |
| 80-89 | 543 | 82.0239 | -0.2012 | 0.7682 | 0.9840 |
| 90+ | 3 | 90.3333 | -2.1908 | 2.1908 | 2.2843 |

### Residuals and large errors

Residuals are defined as prediction minus actual. A positive mean error indicates overall overprediction; a negative value indicates overall underprediction. The largest absolute errors are shown without player names or IDs:

| actual | prediction | residual | absolute_error | age | height_cm | weight_kg |
| --- | --- | --- | --- | --- | --- | --- |
| 49.000 | 58.884 | 9.884 | 9.884 | 24.000 | 184.000 | 76.000 |
| 62.000 | 71.261 | 9.261 | 9.261 | 30.000 | 190.000 | 78.000 |
| 52.000 | 59.987 | 7.987 | 7.987 | 21.000 | 190.000 | 79.000 |
| 62.000 | 69.770 | 7.770 | 7.770 | 23.000 | 187.000 | 93.000 |
| 55.000 | 62.583 | 7.583 | 7.583 | 30.000 | 175.000 | 80.000 |
| 55.000 | 62.537 | 7.537 | 7.537 | 28.000 | 178.000 | 72.000 |
| 54.000 | 61.444 | 7.444 | 7.444 | 25.000 | 182.000 | 77.000 |
| 64.000 | 70.437 | 6.437 | 6.437 | 27.000 | 187.000 | 82.000 |
| 57.000 | 63.009 | 6.009 | 6.009 | 20.000 | 180.000 | 80.000 |
| 64.000 | 70.003 | 6.003 | 6.003 | 29.000 | 175.000 | 70.000 |

![Outfield actual versus predicted](baseline_error_analysis_figures/outfield_actual_vs_predicted.png)
![Outfield residual distribution](baseline_error_analysis_figures/outfield_residual_distribution.png)
![Outfield error by rating range](baseline_error_analysis_figures/outfield_error_by_rating_range.png)

### Feature/error observations

The five largest absolute Spearman associations with absolute error are shown below. These are observational associations within this test set, not causal effects or evidence that a feature should be added, removed, or transformed.

| feature | signed_error_spearman | absolute_error_spearman |
| --- | --- | --- |
| skill_long_passing | -0.0054 | 0.0630 |
| mentality_interceptions | -0.1250 | 0.0603 |
| defending_sliding_tackle | -0.1345 | 0.0567 |
| defending_marking_awareness | -0.1339 | 0.0544 |
| defending_standing_tackle | -0.1271 | 0.0524 |


## Limitations

- This is a single fixed grouped test split, so results describe this split rather than all possible future FIFA data.
- Correlation does not establish causation, feature importance, or a justified modeling change.
- Large-error rows identify where the baseline misses most; they do not by themselves explain why.
- The report does not claim that either population needs a particular model, feature, or hyperparameter change.

## Evidence-based findings

- The exact measured test metrics and residual direction are reported above for each population.
- Rating-range tables show whether observed error levels vary across actual-rating bands and include sample counts to make distribution support visible.
- Large-error tables and association tables identify concrete follow-up questions while preserving the distinction between observation and explanation.

## Implications for future work

Future milestones can use the reported rating bands, large-error cases, and strongest observed associations to define targeted hypotheses and additional validation. Any modeling change should be tested separately with leakage-safe splits and held-out evaluation; these findings alone do not select a redesign.
