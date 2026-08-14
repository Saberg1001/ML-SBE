# Trend classifier report

Best model by grouped CV: **lightgbm**

| model         | status   |   tuning_weighted_macro_f1 |   tuning_doi_macro_f1 |   validation_weighted_macro_f1 |   validation_macro_f1 |   validation_doi_macro_f1 |   validation_balanced_accuracy |   validation_accuracy |   validation_weighted_log_loss |   validation_weighted_ordinal_mae |   validation_weighted_severe_reversal_rate |   best_trial |   elapsed_minutes |
|:--------------|:---------|---------------------------:|----------------------:|-------------------------------:|----------------------:|--------------------------:|-------------------------------:|----------------------:|-------------------------------:|----------------------------------:|-------------------------------------------:|-------------:|------------------:|
| lightgbm      | ok       |                   0.568256 |              0.293368 |                       0.470959 |              0.462636 |                  0.255416 |                       0.459499 |              0.466981 |                       1.14701  |                          0.746872 |                                   0.255307 |           25 |           2.92653 |
| catboost      | ok       |                   0.561169 |              0.288343 |                       0.471408 |              0.458989 |                  0.252922 |                       0.45891  |              0.466981 |                       1.02735  |                          0.727904 |                                   0.254513 |           37 |          24.7838  |
| xgboost       | ok       |                   0.557762 |              0.282633 |                       0.495937 |              0.478242 |                  0.261828 |                       0.479255 |              0.485849 |                       1.11755  |                          0.679592 |                                   0.224423 |           40 |           6.10255 |
| random_forest | ok       |                   0.52965  |              0.268103 |                       0.448293 |              0.413948 |                  0.241248 |                       0.414855 |              0.424528 |                       0.995623 |                          0.791598 |                                   0.29754  |           10 |           6.26118 |

## Figures

- `figures/model_metric_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/optuna_history.png`
- `figures/best_model_feature_importance.png` (when available)
