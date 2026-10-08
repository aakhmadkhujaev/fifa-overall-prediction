# FIFA Player Overall Rating Prediction (V1)

A PyTorch regression project that estimates a FIFA player's overall rating (`overall`) from player attributes, plus a Streamlit app for exploring the final V1 evaluation.

## V1 scope

Included:

- Data loading and validation, feature selection, grouped train/validation/test splitting, and train-only preprocessing.
- Two independent feedforward neural networks: one for goalkeepers and one for outfield players.
- Reproducible experiments (baseline run, learning-rate search, batch-size search, training-duration check) and the fixed final V1 training run.
- A final held-out test evaluation with structured JSON/CSV artifacts.
- A visualization layer (`src/visualization/`) and a Streamlit evaluation explorer (`app.py`).

**Not included in V1:**

- **Live prediction.** The app does not accept player attributes or run a model. It only displays the stored evaluation results and the stored held-out test predictions.
- **A simple non-neural baseline comparison.** This is deferred. The project has not compared the networks with a mean predictor, linear regression, or similar, so it makes no claim that the networks are meaningfully better than such a baseline. Note that the "baseline" training and error-analysis reports refer to the first neural-network configuration (learning rate 0.001), not to a non-neural model.

## Pipeline

```text
raw CSV ─► validation ─► goalkeeper / outfield split ─► feature selection
        ─► grouped 70/15/15 split by player_id (seed 42)
        ─► train-fitted imputation, encoding, scaling
        ─► feedforward network (64 → 32 → 1, ReLU, MSE loss, Adam)
        ─► validation-MAE checkpointing + early stopping
        ─► held-out test evaluation ─► JSON/CSV artifacts ─► Streamlit explorer
```

Preprocessing is fitted on the training split only, and rows from the same `player_id` never appear in more than one split of a population. See `reports/modeling_strategy.md` and `reports/feature_analysis_report.md` for the feature decisions and leakage review.

```text
app.py                       Streamlit app (layout and display only)
src/data/                    dataset loading and validation
src/features/                feature lists and goalkeeper/outfield separation
src/preprocessing/           grouped split and train-fitted preprocessing
src/training/                dataset, model, DataLoaders, trainer
src/experiments/             experiment runner, searches, final V1 training, checkpoint loader
src/evaluation/              error analysis, final V1 evaluation, artifact validation
src/visualization/           artifact loader, accessors, filters, matplotlib figures
tests/                       pytest suite (synthetic data; no dataset or checkpoints needed)
reports/                     written reports; reports/generated/ holds the committed artifacts
```

## Setup

Developed with Python 3.14 and verified on Python 3.13. CPU is sufficient; a GPU is not required.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset location

The raw dataset is not part of the repository. To train or regenerate artifacts, place the FIFA male players (legacy) file at:

```text
data/raw/male_players (legacy).csv
```

`data/raw/` is git-ignored. The app and the test suite do **not** need the dataset or any checkpoint.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app reads only `reports/generated/v1_final_evaluation.json` and `reports/generated/v1_test_predictions.csv`. It has five sections (sidebar): Overview, Performance, Training, Error Analysis, and Test Predictions (a filterable, downloadable table of the stored test predictions). A sidebar selector switches between both populations, goalkeepers, or outfield players.

## Run the tests

```bash
python -m pytest
```

The tests use small synthetic data. They include a trainer → checkpoint → evaluation check that fails if a saved checkpoint's training history disagrees with the training result.

## Regenerate the V1 artifacts

This retrains both V1 models with the fixed configurations in `src/experiments/final_v1.py` (about a minute on a CPU) and needs the dataset. The checkpoints (`models/*.pt`) are git-ignored and are never committed.

```bash
# 1. Keep a copy of the current official checkpoints and training record.
mkdir models/backup_before_regeneration
cp models/*.pt reports/generated/v1_final_results.json models/backup_before_regeneration/

# 2. Retrain, then evaluate the new checkpoints on the held-out test split.
python -m src.experiments.final_v1
python -m src.evaluation.v1_evaluation

# 3. Check the artifacts, comparing predictive metrics with the saved training record.
python -m src.evaluation.v1_validation models/backup_before_regeneration/v1_final_results.json
```

The validation step reports problems if the artifacts contain absolute paths, if the evaluation history disagrees with the training record, if predictive metrics changed, or if prediction row counts, IDs, or feature order are inconsistent. Training is seeded and deterministic on CPU, so the predictive metrics are expected to reproduce exactly.

Generated files record only project-relative paths (for example `models/fifa_overall_goalkeeper_v1.pt`). Checkpoints embed a pickled preprocessing state, so only load checkpoints you produced yourself.

## Metrics

- **MAE** (mean absolute error, rating points) is the primary metric.
- **RMSE** and **R²** are reported alongside it.
- Validation metrics drive checkpoint selection and early stopping; test metrics are computed once per final configuration and are never used to choose a configuration.
- Training loss is recorded per epoch; a training-set MAE is not reported.

## V1 results

Held-out test results (one deterministic grouped split; seed 42). The hyperparameters were chosen using validation MAE only (`reports/learning_rate_search.md`, `reports/batch_size_search.md`, `reports/training_duration_analysis.md`).

| Population | Test rows | Features | Learning rate | Batch size | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Goalkeeper | 2,835 | 12 | 0.01 | 128 | 0.3830 | 0.4837 | 0.9958 |
| Outfield | 21,536 | 35 | 0.00173 | 256 | 0.7248 | 0.9713 | 0.9803 |

Validation results at the selected epoch:

| Population | Best epoch | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: | ---: |
| Goalkeeper | 20 | 0.3790 | 0.4764 | 0.9962 |
| Outfield | 27 | 0.7222 | 0.9743 | 0.9804 |

Training length (maximum 30 epochs, patience 7), from the training record `reports/generated/v1_final_results.json`:

| Population | Epochs run | Best epoch | Early stopping triggered |
| --- | ---: | ---: | --- |
| Goalkeeper | 27 | 20 | Yes |
| Outfield | 30 | 27 | No (ran to the epoch cap) |

The full evaluation, including error direction, rating-range summaries, and the largest errors, is in `reports/final_v1_evaluation.md` and the Error Analysis section of the app.

## Limitations

- Results come from one deterministic grouped held-out split and describe these two fixed test sets. The goalkeeper-versus-outfield difference is descriptive and is not a causal explanation.
- Rating bands with few test samples (for example outfield 90+, 3 rows) should not be generalized.
- The same `player_id` may exist in both populations; goalkeeper and outfield models are trained and evaluated independently (`reports/modeling_strategy.md`).
- The checkpoints are not distributed with the repository; regenerating them requires the dataset.
- Live prediction and the simple non-neural baseline comparison are out of scope for V1, as described above.
