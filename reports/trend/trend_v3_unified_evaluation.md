# Trend-v3 unified evaluation

Generated: 2026-09-14T08:38:04.409773+00:00

Metric convention: `delta = log10(sigma_B) - log10(sigma_A); all errors in log10 units`

Held-out split: `/home/ziyiguo/project/IonConductivity/data/trend/data-trend-v3-pairs-feature-validation.csv`

Rows: 164 before within-DOI dedup, 164 after (0 duplicate pair rows dropped).

## Headline comparison

| candidate | kind | status | weighted MAE | vs zero-change | direction acc | within 2x | delta top-1 | absolute top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_trend_v3` | model | ✅ validated | 0.2471 | **beats** (+0.0235) | 0.4878 | 0.6220 | 0.4000 | 0.2000 |
| `residual_trend_v3` | model | ⚠️ flagged | 0.1369 | **beats** (+0.1337) | 0.8049 | 0.8232 | 0.6000 | 0.2000 |
| `zero_change` | baseline | — | 0.2706 | no (+0.0000) | 0.2683 | 0.6159 | 0.4000 | 0.0000 |
| `group_mean_change` | baseline | — | 0.2477 | **beats** (+0.0229) | 0.3049 | 0.6098 | 0.4000 | 0.0000 |

`vs zero-change` is `zero_change_MAE - model_MAE`; positive is better. A ⚠️ flagged model's held-out numbers are not a trustworthy deployment estimate — see its degeneracy check below.

## Degeneracy checks

### `direct_trend_v3`

Status: **validated_candidate**

Cleared all structural checks.

### `residual_trend_v3`

Status: **experimental_not_for_deployment**

Failing conditions:
- anchor_leakage_suspected (98% of pairs have an endpoint in the anchor model's training data)

Absolute-anchor leakage check:
- either endpoint in anchor training data: 98.2% of pairs
- both endpoints in anchor training data: 74.4% of pairs
- pairs with neither endpoint seen (leak-free): 3 of 164

> A high overlap means d_0 recalls anchor training labels, so the held-out metrics above overstate genuine out-of-sample skill. An honest estimate needs an anchor retrained without the trend-validation materials.

## Full metrics

```json
{
  "direct_trend_v3": {
    "n_pairs": 164,
    "n_dois": 10,
    "weighted_mae_delta_log10_ratio": 0.24710792232777792,
    "mae_delta_log10_ratio": 0.2996654847482919,
    "weighted_rmse_delta_log10_ratio": 0.37157228634723755,
    "rmse_delta_log10_ratio": 0.42759123748260724,
    "weighted_r2_log10_sigma_b": 0.6364327922425144,
    "r2_delta_log10_ratio": 0.17675266394718858,
    "spearman_delta_log10_ratio": 0.4791021935155879,
    "direction_accuracy_at_0.1_log10": 0.4878048780487805,
    "increase_precision_at_0.1_log10": 0.5921052631578947,
    "increase_recall_at_0.1_log10": 0.7258064516129032,
    "decrease_precision_at_0.1_log10": 0.5227272727272727,
    "decrease_recall_at_0.1_log10": 0.39655172413793105,
    "within_two_fold_ratio": 0.6219512195121951,
    "within_five_fold_ratio": 0.8902439024390244,
    "median_absolute_error_delta_log10_ratio": 0.20054598124466555,
    "p90_absolute_error_delta_log10_ratio": 0.7250705028404699,
    "prediction_std_delta_log10_ratio": 0.2842804825482388,
    "group_top1_accuracy": 0.4,
    "group_absolute_top1_accuracy": 0.2
  },
  "residual_trend_v3": {
    "n_pairs": 164,
    "n_dois": 10,
    "weighted_mae_delta_log10_ratio": 0.13689407328626282,
    "mae_delta_log10_ratio": 0.1792328220660716,
    "weighted_rmse_delta_log10_ratio": 0.18165969763985176,
    "rmse_delta_log10_ratio": 0.24417433926664378,
    "weighted_r2_log10_sigma_b": 0.9131009058610063,
    "r2_delta_log10_ratio": 0.8032290970448337,
    "spearman_delta_log10_ratio": 0.8793783370378546,
    "direction_accuracy_at_0.1_log10": 0.8048780487804879,
    "increase_precision_at_0.1_log10": 0.7837837837837838,
    "increase_recall_at_0.1_log10": 0.9354838709677419,
    "decrease_precision_at_0.1_log10": 0.8846153846153846,
    "decrease_recall_at_0.1_log10": 0.7931034482758621,
    "within_two_fold_ratio": 0.823170731707317,
    "within_five_fold_ratio": 0.9817073170731707,
    "median_absolute_error_delta_log10_ratio": 0.12606581864998728,
    "p90_absolute_error_delta_log10_ratio": 0.40201708871295205,
    "prediction_std_delta_log10_ratio": 0.39561330120704863,
    "group_top1_accuracy": 0.6,
    "group_absolute_top1_accuracy": 0.2
  },
  "zero_change": {
    "n_pairs": 164,
    "n_dois": 10,
    "weighted_mae_delta_log10_ratio": 0.27057398610424527,
    "mae_delta_log10_ratio": 0.3360729963111053,
    "weighted_rmse_delta_log10_ratio": 0.4101867899493105,
    "rmse_delta_log10_ratio": 0.48941991524976136,
    "weighted_r2_log10_sigma_b": 0.5569411510312463,
    "r2_delta_log10_ratio": -0.0032450929167755493,
    "spearman_delta_log10_ratio": NaN,
    "direction_accuracy_at_0.1_log10": 0.2682926829268293,
    "increase_precision_at_0.1_log10": 0.0,
    "increase_recall_at_0.1_log10": 0.0,
    "decrease_precision_at_0.1_log10": 0.0,
    "decrease_recall_at_0.1_log10": 0.0,
    "within_two_fold_ratio": 0.6158536585365854,
    "within_five_fold_ratio": 0.9024390243902439,
    "median_absolute_error_delta_log10_ratio": 0.23648264747425224,
    "p90_absolute_error_delta_log10_ratio": 0.673721123026792,
    "prediction_std_delta_log10_ratio": 0.0,
    "group_top1_accuracy": 0.4,
    "group_absolute_top1_accuracy": 0.0
  },
  "group_mean_change": {
    "n_pairs": 164,
    "n_dois": 10,
    "weighted_mae_delta_log10_ratio": 0.24771206889576497,
    "mae_delta_log10_ratio": 0.3285017714019244,
    "weighted_rmse_delta_log10_ratio": 0.3736879963344758,
    "rmse_delta_log10_ratio": 0.4777077309396599,
    "weighted_r2_log10_sigma_b": 0.632280745937031,
    "r2_delta_log10_ratio": 0.16735093302310278,
    "spearman_delta_log10_ratio": 0.16247938931460257,
    "direction_accuracy_at_0.1_log10": 0.3048780487804878,
    "increase_precision_at_0.1_log10": 0.6666666666666666,
    "increase_recall_at_0.1_log10": 0.03225806451612903,
    "decrease_precision_at_0.1_log10": 0.6,
    "decrease_recall_at_0.1_log10": 0.10344827586206896,
    "within_two_fold_ratio": 0.6097560975609756,
    "within_five_fold_ratio": 0.8841463414634146,
    "median_absolute_error_delta_log10_ratio": 0.22796597782486236,
    "p90_absolute_error_delta_log10_ratio": 0.746488371921727,
    "prediction_std_delta_log10_ratio": 0.1041227857944844,
    "group_top1_accuracy": 0.4,
    "group_absolute_top1_accuracy": 0.0
  }
}
```

## Verdict

1 model(s) cleared all structural checks: ['direct_trend_v3']
