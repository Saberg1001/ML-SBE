# data-absolute-v2 数据质量检查

## 文件用途

`data-absolute-v2-family-audit.csv` 不是训练全集。当前重建后有254条，记录原 `family=unknown` 行的标签推荐依据。完整来源表是 `data-absolute-v2.csv`，共1187条。

当前特征流程以 `1e-6 S/cm` 为最低阈值。当前文件有300条低于阈值，剩余887条模型候选。v2来源合并和模型清洗均将Liverpool明确温度限制为20至30°C。

## 检查结果

- 1187条、1078个唯一标准化化学式。
- 72个同式重复组，共181条；全部来自不同DOI，没有残留“同式+同DOI”重复。
- 阈值后仍有61个同式重复组，共156条；其中57组具有不同电导率。
- 剩余最大跨度为0.9985 log10，11组跨度大于0.5 log10。
- 没有重复ID、缺失DOI、缺失标准化化学式、非正电导率或大于1 S/cm的记录。
- `garnet_like` 已全局归一为 `garnet`；LiNbCl4O和LiTaCl4O的宽泛 `halides` 标签按组成统一为 `oxyhalides`。
- 3个多结构冲突组按指定目标 family 解决：Li7P3S11选择 `thio_lisicon`，Li10Sn(PS6)2选择 `lgps`，Li1.6Al0.6Ge1.4P3O12选择 `nasicon`。
- 仅显式删除2条已复核的Caltech高温外推记录：`caltech_icsd_65051`（418 K）和 `caltech_icsd_100169`（473 K）；不按温度阈值扩大删除其他Caltech记录。
- 删除后 `Li2.08Ti0.92In1.08P3O12` 不再进入数据；`Li14Zn(GeO4)4` 仍保留独立的Liverpool 25°C实测记录（`1e-6 S/cm`）。
- Liverpool原始候选中374条因不在20至30°C而排除；DOI和化学式去重后，当前主表有231条Liverpool记录。
- 人工文献补录 `a71299`：`Li2.5Y0.75W0.25Cl3.5Br2.5`，高能球磨法，`6.38e-3 S/cm`，DOI `10.1002/aenm.71299`。该化学式与实验评估点 `hal_021` 重叠，严格外部评估时必须排除该实验点。

## 清洗候选版

生成 `data-absolute-v2-model-clean.csv`，不覆盖原始v2：

- 过滤300条 `<1e-6 S/cm` 记录；Liverpool 20至30°C规则已在上游v2重建时执行。
- 对重复公式组保留文献报道的最高电导率记录，使一个化学式只对应一个训练目标。
- 电导率平局时依次优先人工检查v1、明确室温Liverpool、Caltech。
- 移除95条同式冗余记录。
- 最终792条，对应792个唯一标准化化学式。
- 792条均可成功生成35个模型特征，无特征解析失败。
- family 别名已统一；3个多结构冲突已按目标结构解决，高温外推记录不进入清洗数据。

采用最高文献报道值与此前大跨度冲突处理保持一致。所有未保留的文献值仍记录在公式审计和排除表中。

## 输出

- `data/absolute/data-absolute-v2-model-clean.csv`：一式一值模型候选数据。
- `data/absolute/data-absolute-v2-model-clean-excluded.csv`：300条阈值排除和95条公式合并记录。
- `data/absolute/data-absolute-v2-model-clean-formula-audit.csv`：61个公式组的候选值与保留依据。
- `data/absolute/data-absolute-v2-model-clean-quality-audit.csv`：family冲突和高温外推等复核项。
- `data/absolute/data-absolute-v2-model-clean-summary.json`：机器可读汇总。
