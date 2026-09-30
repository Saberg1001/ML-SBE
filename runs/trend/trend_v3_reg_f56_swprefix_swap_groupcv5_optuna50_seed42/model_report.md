# Trend v3 regressor report

Best model by grouped CV: **lightgbm**

| model    | status   |   tuning_weighted_mae |   validation_weighted_mae |   validation_weighted_rmse |   validation_weighted_r2 |   validation_r2 |   validation_sign_accuracy |   best_trial |   elapsed_minutes |
|:---------|:---------|----------------------:|--------------------------:|---------------------------:|-------------------------:|----------------:|---------------------------:|-------------:|------------------:|
| lightgbm | ok       |              0.336441 |                  0.210275 |                   0.301321 |                 0.458619 |        0.400443 |                   0.707317 |           43 |           2.95841 |

## Figures

- `figures/model_metric_comparison.png`
- `figures/residual_histograms.png`
- `figures/actual_vs_predicted.png`
- `figures/best_model_feature_importance.png` (when available)
