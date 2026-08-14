# Trend classifier report

Best model by fixed validation: **catboost**

| model         | status   |   tuning_weighted_macro_f1 |   tuning_doi_macro_f1 |   validation_weighted_macro_f1 |   validation_macro_f1 |   validation_doi_macro_f1 |   validation_balanced_accuracy |   validation_accuracy |   validation_weighted_log_loss |   validation_weighted_ordinal_mae |   validation_weighted_severe_reversal_rate |   best_trial |   elapsed_minutes |
|:--------------|:---------|---------------------------:|----------------------:|-------------------------------:|----------------------:|--------------------------:|-------------------------------:|----------------------:|-------------------------------:|----------------------------------:|-------------------------------------------:|-------------:|------------------:|
| catboost      | ok       |                   0.639417 |              0.275033 |                       0.639417 |              0.606001 |                  0.275033 |                       0.594676 |              0.596698 |                       0.853336 |                          0.505353 |                                   0.18132  |           42 |          0.600241 |
| random_forest | ok       |                   0.620825 |              0.2741   |                       0.620825 |              0.578761 |                  0.2741   |                       0.578191 |              0.582547 |                       0.853016 |                          0.539054 |                                   0.181714 |           31 |          1.16053  |
| xgboost       | ok       |                   0.616687 |              0.278364 |                       0.616687 |              0.545098 |                  0.278364 |                       0.531946 |              0.554245 |                       0.974392 |                          0.573717 |                                   0.222569 |           44 |          0.268983 |
| lightgbm      | ok       |                   0.60348  |              0.265598 |                       0.60348  |              0.538917 |                  0.265598 |                       0.5228   |              0.523585 |                       0.912359 |                          0.580145 |                                   0.212628 |           22 |          0.155733 |

## Figures

- `figures/model_metric_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/optuna_history.png`
- `figures/best_model_feature_importance.png` (when available)
