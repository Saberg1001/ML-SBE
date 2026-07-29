# IonConductivity

本项目用于基于化学式和手工构造的组成特征预测离子电导率。模型目标为
`log10(Ionic conductivity (S cm-1))`，输出时同时给出 log10 预测值和换算后的
`S cm-1` 电导率。

## 目录结构

```text
config/                 氧化态和离子半径修正规则
rawdata/                原始数据和待预测输入
main/                   数据处理、特征、划分、训练和预测代码
outputs/models/current/ 当前模型输出
outputs/models/legacy/  重构前模型输出（只读保留）
outputs/analyses/       图表和其他分析结果
predictions/            历史和当前预测结果
requirements.txt        Python 依赖
```

## 环境准备

建议在项目专用虚拟环境或 conda 环境中安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括 `pandas`、`scikit-learn`、`pymatgen`、`mendeleev`、`lightgbm`、
`optuna`、`ngboost` 和 `matplotlib`。

## 完整流程

统一 API 位于 `main`：

```python
from main import PipelineConfig, run_training_pipeline

result = run_training_pipeline(PipelineConfig())
print(result.train.output_dir)
```

默认流程依次执行原始数据清理、组成特征生成、训练/测试划分和 Optuna 训练，
模型保存到 `outputs/models/current/`。可通过各阶段的配置对象修改阈值、划分方法、
模型类型和 trial 数。

完整特征表不再单独写入顶层 `features/`。特征生成和数据划分在内存中完成后，
每次训练都会在模型运行目录的 `data/train.csv` 和 `data/test.csv` 中保存完整的
训练/测试特征表，并同时保存实际入模特征列表和缺失值填充信息。

每次训练都会在运行目录的 `figures/` 中强制生成与旧版相同格式和文件名的训练/测试
拟合图、指标表、Top 10 特征重要性占比图及组合汇总图，便于与历史实验直接比较。
残差、Optuna 优化历史和 Top 20 原始重要性图作为额外诊断保留。

预测使用同一套特征实现：

```python
from main import PredictConfig, predict_formulas

result = predict_formulas(
    "rawdata/experimental-data",
    "outputs/models/current/<run_name>",
    PredictConfig(
        dataset_name="experimental-data",
        prediction_purpose="Evaluate predictions against experimental conductivity",
    ),
)
```

预测默认按模型和数据集保存到稳定路径：

```text
predictions/by_model/<training_run>/<model_name>/<dataset_name>/
├── predictions.csv
├── features.csv
├── evaluation.json
└── run_metadata.json
```

相同训练运行、模型和数据集再次预测时更新同一目录，不按时间重复保存。
`run_metadata.json` 说明模型、输入数据和预测用途；`evaluation.json` 在输入含有
`reference_mS_cm` 时记录误差及排序指标，否则明确记录缺少真实值、无法评估。

## 特征工程

训练和预测统一调用 `main.features`：

- 解析电导率并转换为 `log10_conductivity`
- 按配置过滤电导率阈值
- 清理有机样本和电荷异常样本
- 添加若干交互特征
- 添加 8 个 Small/Kong 组成特征
- 编码材料 Family
- 使用训练集特征中位数填补缺失值

特征计算依赖 `config/oxidation_states.json` 和
`config/ionic_radius_overrides.json`。

## 历史结果

`outputs/models/legacy/` 和 `predictions/` 中重构前生成的模型及预测结果继续保留，供结果复核；
新代码不会向这些旧模型目录写入内容。

## 输出说明

模型目录中常见文件：

```text
model_comparison.csv            模型对比结果
lightgbm/final_results.json     LightGBM 训练参数与指标
lightgbm/model.joblib           可用于预测的模型文件
lightgbm/test_predictions.csv   测试集预测
lightgbm/feature_importance.csv 特征重要性
feature_schema.json             模型特征与 Family 编码
summary.json                    训练摘要和最佳模型
```

预测结果中常见列：

- `ID`
- `True Composition`
- `Family`
- `model_name`
- `pred_log10_conductivity`
- `pred_conductivity_S_cm-1`
