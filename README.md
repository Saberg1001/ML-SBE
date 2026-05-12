# IonConductivity

本项目用于基于化学式和手工构造的组成特征预测离子电导率。模型目标为
`log10(Ionic conductivity (S cm-1))`，输出时同时给出 log10 预测值和换算后的
`S cm-1` 电导率。

## 当前推荐模型

目前项目中效果最好的模型是：

```text
models/outputs_optuna/ionic_26_features_random_gt1e-6_50
```

该模型为 26 特征、随机划分、`Ionic conductivity > 1e-6 S cm-1` 数据集上的
LightGBM + Optuna 50 次 trial 版本。当前预测脚本
`predict_internal_formulas.py` 的默认模型目录已经指向这个路径，因此不额外指定
`--model-dir` 时，预测会默认使用该模型。

关键模型文件：

```text
models/outputs_optuna/ionic_26_features_random_gt1e-6_50/lightgbm/model.joblib
```

当前记录的主要指标：

- 5-fold CV MAE: `0.4863`
- Test MAE: `0.4698`
- Test RMSE: `0.6591`
- Test R2: `0.5400`

## 目录结构

```text
config/                 氧化态和离子半径修正规则
rawdata/                原始数据和待预测输入
features/               已生成的特征表
get_feature/            特征生成、筛选和划分脚本
models/                 特征工程、训练脚本和模型输出
predictions/            预测结果
predict_internal_formulas.py  公式预测入口
requirements.txt        Python 依赖
```

## 环境准备

建议在项目专用虚拟环境或 conda 环境中安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括 `pandas`、`scikit-learn`、`pymatgen`、`mendeleev`、`lightgbm`、
`optuna`、`ngboost` 和 `matplotlib`。

## 使用当前最佳模型预测

默认输入文件为：

```text
rawdata/expriment-test
```

默认输出文件为：

```text
predictions/expriment-test_predictions.csv
```

直接运行：

```bash
python predict_internal_formulas.py
```

等价于显式指定当前最佳模型：

```bash
python predict_internal_formulas.py \
  --input rawdata/expriment-test \
  --model-dir models/outputs_optuna/ionic_26_features_random_gt1e-6_50 \
  --output predictions/expriment-test_predictions.csv
```

预测脚本会同时写出：

- 预测结果 CSV：包含 `pred_log10_conductivity` 和 `pred_conductivity_S_cm-1`
- 特征表 CSV：默认保存为 `*_features.csv`
- 排名指标 JSON：默认保存为 `*_metrics.json`

输入可以是纯公式列表，也可以是 CSV/TSV。带表头时脚本会自动识别常见列名：
`True Composition`、`formula`、`composition`、`ID`、`conductivity` 等。

常用参数：

```bash
python predict_internal_formulas.py \
  --input path/to/formulas.csv \
  --output predictions/my_predictions.csv \
  --formula-column "True Composition" \
  --id-column ID
```

默认会跳过同时含 C 和 H 的有机样公式；如需预测这类公式，可添加：

```bash
--allow-organic
```

## 训练模型

Optuna 训练入口为：

```bash
python models/train_models_optuna.py
```

复现当前推荐模型对应的数据配置和 trial 数：

```bash
python models/train_models_optuna.py \
  --model lightgbm \
  --train features/ionic_26_features_random_gt1e-6_train.csv \
  --test features/ionic_26_features_random_gt1e-6_test.csv \
  --n-trials 50
```

训练结果会保存到：

```text
models/outputs_optuna/<feature_set_name>/
```

其中 LightGBM 模型位于：

```text
models/outputs_optuna/<feature_set_name>/lightgbm/model.joblib
```

## 特征工程

训练和预测都会调用 `models/feature_engineering.py` 中的特征处理逻辑：

- 解析电导率并转换为 `log10_conductivity`
- 对上限值样本设置替代值和样本权重
- 删除冗余或弱相关特征
- 添加若干交互特征
- 使用训练集特征中位数填补缺失值

基础组成特征由 `get_feature/get_feature.py` 生成，依赖 `config/oxidation_states.json`
和 `config/ionic_radius_overrides.json`。

## 输出说明

模型目录中常见文件：

```text
model_comparison.json/csv       模型对比结果
lightgbm/final_results.json     LightGBM 训练参数与指标
lightgbm/model.joblib           可用于预测的模型文件
lightgbm/test_predictions.csv   测试集预测
lightgbm/feature_importance.csv 特征重要性
data/feature_list.txt           模型使用的最终特征列表
figures/                        训练和评估图
```

预测结果中常见列：

- `ID`
- `True Composition`
- `status`
- `message`
- `charge_residual`
- `pred_log10_conductivity`
- `pred_conductivity_S_cm-1`
- `n_missing_features_filled`

`status` 为 `ok` 表示正常预测；`skipped` 或 `error` 表示该行未生成有效预测。
