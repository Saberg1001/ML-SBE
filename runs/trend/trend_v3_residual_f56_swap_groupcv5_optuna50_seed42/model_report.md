# Residual-correction trend-v3 regression

Architecture: `delta_hat = f_abs(B) - f_abs(A) + h(features)`

Absolute model: runs/absolute/abs_v2_f37_native_family_lgbm_trials50_seed42

## CV metrics
- `weighted_mae_delta_log10_ratio` (mean across 5 folds): 0.2659

## Validation metrics
- `weighted_mae_delta_log10_ratio`: 0.1369
- `mae_delta_log10_ratio`: 0.1792
- `weighted_rmse_delta_log10_ratio`: 0.1817
- `r2_log10_sigma_b`: 0.9131
- `spearman_delta_log10_ratio`: 0.8794
- `direction_accuracy_at_0.1_log10`: 0.8049
- `within_two_fold_ratio`: 0.8232
- `increase_precision_at_0.1_log10`: 0.7838
- `increase_recall_at_0.1_log10`: 0.9355
- `group_top1_accuracy`: 0.6000
- `median_absolute_error_delta_log10_ratio`: 0.1261
- `p90_absolute_error_delta_log10_ratio`: 0.4020

## Validation zero-change baseline
- `weighted_mae_delta_log10_ratio`: 0.2706

Beats zero-change baseline: **True**
Model status: **validated_candidate**
