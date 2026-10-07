# FIFA Overall Rating V1 Evaluation

## Objective

Produce two CPU-compatible V1 production checkpoints for goalkeeper and outfield players using the approved fixed configurations and existing grouped preprocessing/training pipeline.

## Final Configuration

- Goalkeeper: learning rate 0.01, batch size 128, 30 epochs, patience 7, seed 42, CPU, 64/32 hidden sizes.
- Outfield: learning rate 0.0017320508075688787, batch size 256, 30 epochs, patience 7, seed 42, CPU, 64/32 hidden sizes.

## Evaluation Protocol

The dataset is validated before use, split into goalkeeper and outfield populations, and processed with a training-fitted `Preprocessor` using grouped train/validation/test splits. The restored best-validation-MAE model is evaluated on validation and held-out test data with MAE, RMSE, and R2.

## Expected Artifact Structure

- `models/fifa_overall_goalkeeper_v1.pt`
- `models/fifa_overall_outfield_v1.pt`
- `reports/generated/v1_final_results.json`

Each checkpoint contains the model state, optimizer state, training metadata, fixed experiment configuration, ordered feature names, target/group columns, preprocessing schema, and serialized fitted preprocessor state.

## Measured Results

Results will be populated after the approved final V1 training run.

### Goalkeeper

- Validation metrics: pending final training.
- Test metrics: pending final training.
- Best epoch and duration: pending final training.

### Outfield

- Validation metrics: pending final training.
- Test metrics: pending final training.
- Best epoch and duration: pending final training.
