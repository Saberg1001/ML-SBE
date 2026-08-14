# Trend classifier report

Best model by grouped CV: **catboost**

| model         | status   |   tuning_weighted_macro_f1 |   tuning_doi_macro_f1 |   validation_weighted_macro_f1 |   validation_macro_f1 |   validation_doi_macro_f1 |   validation_balanced_accuracy |   validation_accuracy |   validation_weighted_log_loss |   validation_weighted_ordinal_mae |   validation_weighted_severe_reversal_rate |   best_trial |   elapsed_minutes |
|:--------------|:---------|---------------------------:|----------------------:|-------------------------------:|----------------------:|--------------------------:|-------------------------------:|----------------------:|-------------------------------:|----------------------------------:|-------------------------------------------:|-------------:|------------------:|
| catboost      | ok       |                   0.592841 |              0.309076 |                       0.457453 |              0.424752 |                  0.260558 |                       0.425972 |              0.430622 |                        1.12409 |                          0.763735 |                                   0.266331 |           26 |          16.5971  |
| lightgbm      | ok       |                   0.590869 |              0.300134 |                       0.433451 |              0.400726 |                  0.225366 |                       0.400832 |              0.406699 |                        1.70626 |                          0.819892 |                                   0.302833 |           33 |           8.69024 |
| xgboost       | ok       |                   0.570708 |              0.29286  |                       0.489806 |              0.470844 |                  0.262183 |                       0.469522 |              0.473684 |                        1.02815 |                          0.697316 |                                   0.233439 |           10 |           6.86935 |
| random_forest | ok       |                   0.559593 |              0.284332 |                       0.461199 |              0.399591 |                  0.253186 |                       0.388504 |              0.392344 |                        1.00423 |                          0.81785  |                                   0.313627 |           10 |          20.796   |

## Figures

- `figures/model_metric_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/optuna_history.png`
- `figures/best_model_feature_importance.png` (when available)
