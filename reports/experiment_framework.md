# Experiment Framework

Milestone 6A adds a reusable experiment layer without changing the baseline training entry point or model architecture. `ExperimentConfig` captures the settings for one independent goalkeeper or outfield run, and validates values before execution. The existing `FIFAOverallModel` parameters `hidden_size1` and `hidden_size2` are exposed as configuration fields for future controlled experiments.

`ExperimentRunner` owns orchestration only. It selects the population features, fits the existing `Preprocessor`, creates the existing train/validation/test `DataLoader` objects, seeds before constructing `FIFAOverallModel`, and delegates training to the existing `train_model()` function. It does not contain a second training loop.

`ExperimentResult` stores the configuration, selected validation epoch and metrics, held-out test metrics, and training duration. `train_model()` receives only the train and validation loaders. Because that function restores the model state with the best validation MAE before returning, the runner evaluates the test loader only after validation-based model selection. Test performance is not used to select a model or an experiment.

Future experiments can create an `ExperimentConfig`, pass the relevant population dataframe to `ExperimentRunner.run()`, and compare the resulting structured records. The optional checkpoint path is explicit and defaults to no file, so unit tests and dry runs do not generate artifacts. Defaults remain CPU, float32 through the existing dataset/trainer path, and `num_workers=0`.