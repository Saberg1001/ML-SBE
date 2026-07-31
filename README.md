# IonConductivity

本项目基于化学式和组成描述符研究锂离子电解质电导率。目前可运行模型均为
`log10(Ionic conductivity (S cm-1))` 绝对值回归；趋势分类、排序和
`Δlog10` 回归尚未训练；后续趋势模型必须使用经复核的配对表和分组划分。

## 目录结构

```text
config/
├── chemistry/                 氧化态与离子半径规则
└── taxonomy/                  待建立的Family规范词表

data/
├── obelix/raw/                OBLiX all、文献推荐划分及CIF
├── experimental/             113条实验数据及人工标注
├── prediction_inputs/        无训练标签的待预测配方
└── splits/                    固定ID/分组划分manifest

main/                          清洗、特征、划分、训练和预测代码
runs/                          运行索引、每次模型运行及其预测
reports/                       筛选后的正式表格和图片
archive/                       只读历史模型和旧预测
scripts/                       可复用审计与绘图入口
```

目录约定：`obelix/raw` 和 `experimental/raw` 保留源文件，不直接覆盖；
`data/splits` 只保存样本ID、分组和划分manifest，不复制完整特征表。每个
`runs/<task>/<run_id>` 自带配置、入模数据、模型、预测和诊断图，`reports` 只放
精选汇总，`archive` 只读保留历史材料。

## 数据

OBLiX原始数据位于：

```text
data/obelix/raw/all.csv
data/obelix/raw/official_split/train.csv
data/obelix/raw/official_split/test.csv
data/obelix/raw/cifs/
```

其中 `train.csv/test.csv` 表示文献推荐划分。当前工作区的 `train.csv` 内容呈
二进制或加密状态，文件被原样保留，但在取得可读版本前不能直接交给pandas。

实验数据分为原始表和标注表：

```text
data/experimental/raw/experimental-data.tsv
data/experimental/annotations/experimental-data-labeled.csv
```

legacy block文件只用于人工复核，不能直接视为有效趋势配对。待预测配方位于
`data/prediction_inputs/`，不参与训练。

## 当前绝对值baseline

历史模型索引位于 `runs/absolute/registry.csv`，对应运行目录为：

```text
runs/absolute/abs_v0_f26/
runs/absolute/abs_v0_f27_family_ordinal/
runs/absolute/abs_v0_f34_small8/
runs/absolute/abs_v0_f35_small8_family_ordinal/
```

四者均为历史baseline，使用相同的旧随机344/86划分。27和35特征模型的Family
采用连续浮点编码；四个模型的随机划分均存在公式和DOI泄漏。因此这些结果可用于
复盘特征变化，但不能作为趋势模型的最终性能结论。

每个运行内部统一保存：

```text
<run_id>/
├── manifest.json
├── config.json
├── data/
├── lightgbm/
├── predictions/
└── figures/
```

## 使用当前流水线

项目环境：

```text
/home/ziyiguo/miniconda3/envs/IonConductivity
```

完整训练API保持从 `main` 导入：

```python
from main import PipelineConfig, run_training_pipeline

result = run_training_pipeline(PipelineConfig())
print(result.train.output_dir)
```

新训练默认写入 `runs/absolute/`。划分阶段不再额外写顶层 `features/train/test`
副本；每个正式运行在自己的 `data/` 下保存实际入模数据。

预测示例：

```python
from main import PredictConfig, predict_formulas

result = predict_formulas(
    "data/experimental/annotations/experimental-data-labeled.csv",
    "runs/absolute/abs_v0_f26",
    PredictConfig(
        dataset_name="experimental_113",
        prediction_purpose="Evaluate the historical absolute baseline",
    ),
)
```

默认输出与模型运行绑定：

```text
runs/absolute/<run_id>/predictions/<model>/<dataset_id>/
├── predictions.csv
├── features.csv
├── evaluation.json
└── run_metadata.json
```

化学配置位于 `config/chemistry/`。历史模型集中在
`archive/absolute_legacy/models/`，只读保留，不作为代码默认输入。

## Git管理原则

- 跟踪源码、配置、原始/校订数据、manifest和精选报告；
- 不跟踪模型二进制、运行级特征缓存和自动生成训练图；
- 不使用 `latest/current` 表示正式模型，统一使用稳定 `run_id`；
- 数据过滤、Family词表、配对规则或划分变化时创建新版本，不覆盖旧版本。
