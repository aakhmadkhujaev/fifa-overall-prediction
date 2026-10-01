# Engineering Review: Milestone 4.5

**Review date:** 2026-10-01
**Scope:** Existing data loading, validation, feature selection, goalkeeper/outfield separation, grouped preprocessing, and PyTorch data layer.
**Dataset:** `data/raw/male_players (legacy).csv` (161,583 rows, 110 columns)
**Runtime:** Python 3.14.4, pandas 3.0.2, scikit-learn 1.9.0, PyTorch 2.14.0+cpu

No neural network was implemented or trained. The raw dataset was not modified.

## 1. Runtime Safety Findings

### Confirmed safe paths

- `load_dataset` checks that the path exists and is a file, and wraps CSV parsing errors in `DataLoaderError`.
- `select_features` checks every required feature, target, and grouping column before selecting columns.
- `split_gk_and_outfield` uses `player_positions` when available and correctly partitioned the real dataset into 17,970 goalkeepers and 143,613 outfield records.
- `Preprocessor.process` drops rows with missing targets, imputes selected numeric and categorical values, and returns `float32` feature and target frames.
- The real dataset produced non-empty train/validation/test splits for both partitions.

### Findings

- **Medium: empty inputs fail with an indirect scikit-learn error.** `Preprocessor.process` does not explicitly reject an empty frame or a frame that becomes empty after dropping missing targets. The observed error was `ValueError: Found array with 0 sample(s) while a minimum of 1 is required.` This is safe from silent training, but the error does not identify the invalid input or split.
- **Medium: unexpected categorical values can become NaN.** `_encode_categorical` maps only `Right` and `Left`. A synthetic `preferred_foot='Unknown'` value completed preprocessing but produced one non-finite feature value. The real dataset currently contains only `Left` and `Right`, but future files or user-provided data can violate this assumption.
- **Medium: infinite values are not explicitly validated.** Missing-value imputation handles NaN values, but `validate_dataset` checks nulls rather than `np.isfinite`. An input containing `np.inf` reached `StandardScaler`, which raised a scikit-learn error. This prevents a bad tensor in the observed path, but the pipeline has no domain-specific diagnostic.
- **Low: missing required schema is caught only by later functions.** `validate_dataset` reports target presence but does not require `player_id`, position columns, or the V1 feature set. `select_features` and the splitter eventually raise useful errors, but validation does not provide one consolidated schema report.
- **Low: incomplete split dictionaries are silently accepted by `create_dataloaders`.** If a split key is absent from either input dictionary, that DataLoader is omitted instead of raising a clear error. This can allow a later training stage to fail because a loader is missing.
- **Low: a dataset with a position column present but no recognizable goalkeeper values can create an empty goalkeeper partition.** The current real data has valid positions, but `split_gk_and_outfield` does not assert that both partitions are non-empty.

Invalid paths and missing files are covered by tests and behaved as documented.

## 2. Data Leakage Findings

### Verified controls

- `select_features` uses explicit V1 feature lists and appends only `overall` and `player_id` for downstream target/group handling. `overall` is not an input feature.
- `Preprocessor.process` removes `target_col` and `group_col` before creating feature columns.
- Numeric and categorical imputers are fitted on `X_train` only. Validation and test data use `.transform` with those fitted imputers.
- The scaler is fitted on `X_train` only. Validation and test data use the training-fitted scaler.
- The real benchmark found zero player-group overlaps:
  - Goalkeeper: train/validation 0, train/test 0, validation/test 0.
  - Outfield: train/validation 0, train/test 0, validation/test 0.
- The explicit excluded features from the modeling strategy, including `potential`, financial fields, aggregate stats, identifiers, and metadata, were not reintroduced by the current feature lists.

### Residual risk

The guarantees are implemented by convention and tests rather than by a final invariant check. A future caller could pass a custom frame or change feature lists without a direct assertion that `overall` is absent, that all split groups are disjoint, or that transformed values are finite. This is a **Low** maintainability risk for the current milestone, not an observed leakage failure.

## 3. Performance Measurements

The benchmark used `time.perf_counter()` and temporary in-memory code. It did not write to or modify the CSV. The preprocessing stages were measured separately for goalkeeper and outfield data.

| Stage | Goalkeeper | Outfield |
|---|---:|---:|
| CSV loading | 5.590212 s | shared |
| Dataset validation | 0.816898 s | shared |
| Goalkeeper/outfield separation | 0.381162 s | shared |
| Feature selection | 0.015781 s | 0.024267 s |
| Grouped train/validation/test split | 0.061747 s | 0.102710 s |
| Training preprocessing fit and train transform | 0.048430 s | 0.560657 s |
| Validation/test transform | 0.018925 s | 0.079432 s |
| Tensor conversion and Dataset creation | 0.046860 s | 0.012145 s |
| DataLoader creation | 0.001102 s | 0.000112 s |
| Fetching three batches per split | 0.049607 s | 0.018051 s |

The real dataset had 17,970 goalkeeper rows and 143,613 outfield rows. The measured pipeline stages after loading are small relative to CSV parsing. No optimization is justified solely for speed at this scale.

## 4. Memory Considerations

Measured pandas object footprints from the benchmark:

- Raw loaded DataFrame: approximately **185.15 MB**.
- Goalkeeper and outfield partition copies: approximately **20.66 MB** and **166.48 MB**.
- Selected feature frames: approximately **2.14 MB** and **42.30 MB**.
- Processed feature frames: approximately **0.96 MB** and **20.27 MB**.
- PyTorch tensors for all three splits: approximately **0.89 MB** and **19.72 MB**.

The implementation uses `.copy()` during filtering, feature selection, grouped splitting, and preprocessing. These copies are understandable because later operations mutate split frames, and the measured footprints remain well below the available 8 GB RAM. During the split and preprocessing steps, several versions can coexist, so the peak is higher than any one table's size; it is still expected to remain comfortably below 1 GB for this dataset under the measured schema.

The main memory consideration is `pd.read_csv(..., low_memory=False)`, which favors stable type inference at the cost of parser working memory. The current measured raw DataFrame is not large enough to justify changing this behavior without a memory-profile result. PyTorch tensor copies are also small: approximately 20 MB for the larger outfield partition.

There is no repeated dataset load in the current pipeline. DataLoader construction uses `num_workers=0`, which is CPU-compatible and avoids worker-process copies.

## 5. PyTorch Compatibility Findings

The real-data integration path passed for both partitions:

- Features were numeric after imputation and categorical encoding.
- Processed features and targets were `float32`.
- No NaN or infinite values were present in either partition after normal preprocessing.
- Feature dimensions were consistent within each partition: 12 goalkeeper features and 35 outfield features.
- Representative batches had shapes `(256, 12)` and `(256, 1)` for goalkeepers, and `(256, 35)` and `(256, 1)` for outfield players.
- Batch dtypes were `torch.float32` for both features and targets.
- Training uses a random sampler; validation and test use sequential samplers.

The Dataset implementation converts already-processed inputs and does not perform preprocessing. This matches the approved architecture. The unexpected-category probe is the only observed path that can create a non-finite value before tensor conversion.

## 6. Code-Quality Findings

- The source is modular and follows the documented separation between loading, validation, feature selection, preprocessing, and PyTorch data loading.
- The data-loader path and Dataset have useful type annotations, but the public preprocessing and feature functions could use more precise pandas/numpy types and explicit validation contracts.
- `src/training/dataloader.py` has duplicate `typing` imports and imports that are primarily present for annotations. This is minor cleanup, not a runtime issue.
- Split ratios and the random seed are embedded in `Preprocessor.process` rather than exposed as configuration. This is acceptable for V1 but limits reuse and experiment control.
- `Preprocessor.is_goalkeeper` is stored but does not control feature selection; callers must provide the correctly selected frame. The current architecture is consistent, but the relationship should be documented or validated before training is added.
- The empty `train.py`, `evaluate.py`, `predictor.py`, and `config.py` are expected at this milestone and are not defects in the data pipeline.
- The test named `test_preprocessing_leakage` contains no assertions and ends with `pass`. Leakage is partially covered by code inspection and the integration behavior, but this test does not independently detect a regression.
- Tests do not currently cover empty inputs, unexpected categorical values, infinite values, missing schema, or actual group-disjointness assertions.
- There is no project packaging or pytest configuration. In a clean shell with `PYTHONPATH` removed, `pytest -q` failed collection with four `ModuleNotFoundError: No module named 'src'` errors. With `PYTHONPATH=.` configured, all 19 tests passed.

## 7. Issues Ranked

### Critical

None observed.

### High

None observed on the current real dataset and CPU runtime.

### Medium

1. Empty inputs or empty post-target-drop frames fail indirectly inside scikit-learn.
2. Unknown `preferred_foot` values silently become NaN after encoding.
3. Infinite values are not explicitly diagnosed before preprocessing or tensor conversion.
4. Missing or empty goalkeeper/outfield partitions are not explicitly rejected.

### Low

1. `validate_dataset` does not perform complete required-schema or finite-value validation.
2. `create_dataloaders` silently omits incomplete splits.
3. Group-disjointness and target-exclusion guarantees are not asserted as runtime invariants.
4. The leakage test is currently a placeholder with no assertions.
5. Project tests depend on manually setting `PYTHONPATH` in a clean shell.
6. Minor duplicate imports and limited configuration exposure reduce polish but do not affect current behavior.

## 8. Recommended Fixes

1. Add explicit precondition checks in preprocessing for empty input, empty post-filter partitions, and empty train/validation/test splits. Raise project-specific errors with the affected stage and partition.
2. Define a policy for unknown categorical values. Reject them with a clear error or map them deliberately; do not allow an unmapped value to become NaN silently.
3. Validate finiteness for selected features and targets before conversion to PyTorch, with an error naming the affected columns and split.
4. Add schema and invariant checks for required columns, non-empty partitions, disjoint `player_id` sets, target exclusion, consistent feature dimensions, and finite transformed values.
5. Replace the placeholder leakage test with assertions that compare fitted preprocessing state and verify group disjointness on synthetic data.
6. Configure the project without requiring a manual environment variable. The smallest clean fix is a project `pyproject.toml` with pytest `pythonpath = ["."]`; a fuller installable-package configuration can be introduced before deployment. This review does not apply that change because it is outside the requested audit artifact.
7. Do not optimize the current DataFrame or tensor copies yet. Benchmark results show the current pipeline is fast enough and uses a small fraction of 8 GB RAM; revisit only if larger datasets or measured peak-RSS data justify it.

## Test Results

- Exact `pytest -q` in a clean environment without `PYTHONPATH`: collection failed with 4 import errors.
- `PYTHONPATH=.; pytest -q`: **19 passed**.
- No tests or source files were modified for this audit.
