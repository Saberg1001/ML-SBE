# Trend classifier report

Best model by grouped CV: **lightgbm**

| model         | status   |   tuning_weighted_macro_f1 |   tuning_doi_macro_f1 |   validation_weighted_macro_f1 |   validation_macro_f1 |   validation_doi_macro_f1 |   validation_balanced_accuracy |   validation_accuracy |   validation_weighted_log_loss |   validation_weighted_ordinal_mae |   validation_weighted_severe_reversal_rate |   best_trial |   elapsed_minutes |
|:--------------|:---------|---------------------------:|----------------------:|-------------------------------:|----------------------:|--------------------------:|-------------------------------:|----------------------:|-------------------------------:|----------------------------------:|-------------------------------------------:|-------------:|------------------:|
| lightgbm      | ok       |                   0.573978 |              0.293124 |                       0.460129 |              0.435854 |                  0.245301 |                       0.438552 |              0.448113 |                       1.00775  |                          0.760646 |                                   0.27254  |           33 |           3.52664 |
| catboost      | ok       |                   0.562011 |              0.285728 |                       0.435316 |              0.418082 |                  0.24312  |                       0.419096 |              0.429245 |                       0.973878 |                          0.814015 |                                   0.309473 |           35 |          11.2082  |
| random_forest | ok       |                   0.538017 |              0.272014 |                       0.439351 |              0.402042 |                  0.229811 |                       0.405049 |              0.415094 |                       0.993707 |                          0.791764 |                                   0.291356 |           32 |           9.03989 |
| xgboost       | ok       |                   0.534248 |              0.272357 |                       0.467037 |              0.447073 |                  0.244942 |                       0.444561 |              0.45283  |                       1.18125  |                          0.734881 |                                   0.264434 |           43 |           8.26135 |

## Figures

- `figures/model_metric_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/optuna_history.png`
- `figures/best_model_feature_importance.png` (when available)
