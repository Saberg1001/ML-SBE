# Trend classifier report

Best model by fixed validation: **catboost**

| model         | status   |   tuning_weighted_macro_f1 |   tuning_doi_macro_f1 |   validation_weighted_macro_f1 |   validation_macro_f1 |   validation_doi_macro_f1 |   validation_balanced_accuracy |   validation_accuracy |   validation_weighted_log_loss |   validation_weighted_ordinal_mae |   validation_weighted_severe_reversal_rate |   best_trial |   elapsed_minutes |
|:--------------|:---------|---------------------------:|----------------------:|-------------------------------:|----------------------:|--------------------------:|-------------------------------:|----------------------:|-------------------------------:|----------------------------------:|-------------------------------------------:|-------------:|------------------:|
| catboost      | ok       |                   0.523227 |              0.289072 |                       0.523227 |              0.414711 |                  0.289072 |                       0.431076 |              0.443396 |                       0.865803 |                          0.587825 |                                   0.184473 |           10 |          4.51192  |
| lightgbm      | ok       |                   0.512127 |              0.275656 |                       0.512127 |              0.434636 |                  0.275656 |                       0.446854 |              0.457547 |                       0.911088 |                          0.574939 |                                   0.161364 |           28 |          0.679867 |
| xgboost       | ok       |                   0.499739 |              0.270011 |                       0.499739 |              0.454797 |                  0.270011 |                       0.453935 |              0.462264 |                       0.929033 |                          0.710841 |                                   0.259076 |           26 |          1.58739  |
| random_forest | ok       |                   0.491462 |              0.258715 |                       0.491462 |              0.449536 |                  0.258715 |                       0.44837  |              0.457547 |                       0.903024 |                          0.748936 |                                   0.284076 |           32 |          3.75541  |

## Figures

- `figures/model_metric_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/optuna_history.png`
- `figures/best_model_feature_importance.png` (when available)
