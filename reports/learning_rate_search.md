# Learning-Rate Search

Milestone 6B adds a reusable two-stage learning-rate search without running a real FIFA search. It composes with `ExperimentRunner`, so preprocessing, model construction, training, validation, test evaluation, and reproducibility remain owned by the existing experiment pipeline.

## Search Stages

The coarse stage generates deterministic candidates with logarithmic spacing between the configured minimum and maximum. This is appropriate because learning rates are multiplicative quantities: a step from `1e-4` to `2e-4` is more comparable to a step from `1e-3` to `2e-3` than equal-width linear steps would be.

The fine stage identifies the coarse candidate with the lowest validation MAE, then searches a denser logarithmic region around that candidate. The region is clipped to the configured global bounds and controlled by `refinement_factor`. Before execution, fine candidates already evaluated during the coarse stage, or repeated within the fine stage, are removed while preserving candidate order. The reported trial count therefore reflects the number of unique experiments actually executed.

## Selection And Isolation

Candidates are ranked only by validation MAE. Test MAE, RMSE, and R2 remain available in each `ExperimentResult`, but the search never reads them when selecting either the coarse winner or the final winner. The existing runner continues to evaluate the test split only after the model has been selected by validation MAE.

Each candidate receives a new frozen `ExperimentConfig` with only `learning_rate` changed. The search validates that all other Milestone 6B settings remain fixed: seed 42, batch size 256, 30 epochs, patience 7, CPU, zero workers, and the existing 64 -> 32 -> 1 architecture. Goalkeeper and outfield searches use the same implementation with independent fixed population configurations.

The reported best learning rate is the best **tested** candidate in the coarse-plus-fine set. It is not a claim about a mathematically exact or globally optimal learning rate. No real experiment results are included in this report.