# FIFA Overall Rating V1 Evaluation

## Objective

Produce two CPU-compatible V1 production checkpoints for goalkeeper and outfield players using the approved fixed configurations and existing grouped preprocessing/training pipeline.

## Final Configuration

- Goalkeeper: learning rate 0.01, batch size 128, 30 epochs, patience 7, seed 42, CPU, 64/32 hidden sizes.
- Outfield: learning rate 0.0017320508075688787, batch size 256, 30 epochs, patience 7, seed 42, CPU, 64/32 hidden sizes.

Dataset size: 161,583 rows. The goalkeeper population contains 17,970 rows and the outfield population contains 143,613 rows.

## Evaluation Protocol

The dataset is validated before use, split into goalkeeper and outfield populations, and processed with a training-fitted `Preprocessor` using grouped train/validation/test splits. The restored best-validation-MAE model is evaluated on validation and held-out test data with MAE, RMSE, and R2.

## Expected Artifact Structure

- `models/fifa_overall_goalkeeper_v1.pt`
- `models/fifa_overall_outfield_v1.pt`
- `reports/generated/v1_final_results.json`

Each checkpoint contains the model state, optimizer state, training metadata, fixed experiment configuration, ordered feature names, target/group columns, preprocessing schema, and serialized fitted preprocessor state.

## Measured Results

### Goalkeeper

- Best epoch: 20 of 30; early stopping occurred after 27 completed epochs.
- Validation: MAE 0.3790021240711212, RMSE 0.4763753414154053, R2 0.9961926257237792.
- Test: MAE 0.38301607966423035, RMSE 0.48369768261909485, R2 0.9957777825184166.
- Training duration: 11.704663900018204 seconds.

### Outfield

- Best epoch: 27 of 30; early stopping did not occur; all 30 epochs completed.
- Validation: MAE 0.7221816778182983, RMSE 0.9742979407310486, R2 0.9803649429231882.
- Test: MAE 0.7247869372367859, RMSE 0.9712936282157898, R2 0.9802714977413416.
- Training duration: 52.20506209996529 seconds.

Overall orchestration runtime: 65.27050280000549 seconds.

## Comparison With Milestones 6C and 6D

The V1 configurations match the batch sizes selected by Milestone 6C: 128 for goalkeeper and 256 for outfield. V1 validation and test metrics, selected best epochs, and 6D's 50-epoch measured metrics are identical for both populations. The V1 run uses the approved 30-epoch production limit; early stopping selected the same validation-MAE checkpoints.

## Final V1 Conclusions

Both official V1 checkpoints were produced with the approved CPU configurations and validated by reconstructing the model and fitted preprocessor from checkpoint metadata. Deterministic CPU inference reproduced the saved test metrics for both populations. The results support publishing these two checkpoints as the V1 artifacts under the existing validation protocol.
