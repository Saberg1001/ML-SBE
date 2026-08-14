# 成对电导率趋势预测：35 特征迁移建议与相近工作汇总

更新日期：2026-08-06

## 1. 结论先行

原绝对值模型的 35 个特征不应原样复制到成对趋势模型。推荐把每个数值特征从“单个材料描述”转换为以下成对表示：

\[
\Delta f=f_B-f_A,\qquad
|\Delta f|,\qquad
\bar f=(f_A+f_B)/2
\]

对严格为正且跨数量级的特征，可再计算：

\[
\Delta\log f=\log\frac{f_B+\epsilon}{f_A+\epsilon}
\]

其中：

- `Δf` 表示从 A 到 B 的变化方向，是预测“增大/减少”的核心。
- `|Δf|` 表示改动幅度，有助于识别“几乎不变”。
- `mean_f` 表示宿主背景；同一种掺杂变化在不同基体中的作用不一定相同。
- `family` 不计算差值。它在同一配对组内相同，应作为类别上下文，而不是任意整数序数。

首轮不建议直接构造全部 100 多个特征。当前 2539 对数据实际只来自 129 个组、124 个 DOI，独立信息量远小于 pair 数量。建议先训练约 35–50 个特征的 compact 版本，再通过严格的 DOI 分组交叉验证扩展。

## 2. 当前数据与原模型基线

### 2.1 当前趋势数据

以当前工作区文件为准：

| 项目 | 数量 |
|---|---:|
| 原始记录 | 721 |
| clean 记录 | 677 |
| 配对数 | 2539 |
| 配对组 | 129 |
| DOI | 124 |
| 增大 | 875 |
| 减少 | 888 |
| 几乎不变 | 776 |

配对并不独立。最大组包含 30 条原始记录，组合成 435 对；如果随机拆分 pair，同一 DOI、同一组甚至同一材料会同时进入训练集和测试集，结果会明显偏高。

### 2.2 原 35 特征绝对值模型

原 LightGBM 使用 344 条训练数据和 86 条测试数据：

| 指标 | 训练集 | 测试集 |
|---|---:|---:|
| MAE，log10(S/cm) | 0.1068 | 0.4293 |
| RMSE，log10(S/cm) | 0.2349 | 0.6069 |
| R² | 0.9477 | 0.6899 |

这些数值只能视为历史随机划分结果，不能视为独立 DOI 泛化性能。复核发现：

- 41 个 DOI 同时出现在训练集和测试集。
- 86 条测试记录中有 75 条来自训练阶段已经出现的 DOI。
- 17 条测试记录与训练记录具有完全相同的 35 维特征向量。

因此测试 MAE 和原特征重要性都可能偏乐观。趋势任务中必须使用 DOI 分组切分；以下重要性只作为弱先验，而不是选特征的证据终点。

另一个重要结果是：

| 特征版本 | 测试 MAE | 测试 R² |
|---|---:|---:|
| 34 个组成特征 | 0.4171 | 0.7078 |
| 35 个特征，增加 ordinal `family` | 0.4293 | 0.6899 |

因此，当前 `family=0,1,2...` 的序数编码没有带来提升。趋势模型应改为 one-hot、CatBoost 原生类别处理或分 family 校准。

### 2.3 绝对值任务和趋势任务的特征重要性为什么会不同

二者预测的是不同的统计问题：

- 绝对值模型近似学习 `logσ = family 基线 + 组成 + 结构 + 工艺`。
- 当前趋势模型学习的是 `同一 DOI/温度/制备/family 条件下，B 相对 A 的变化`。

对当前文件做方差分解可见；趋势侧使用实际进入配对的 647 条唯一 `group_id + ID` 记录：

| 结果 | 数值 |
|---|---:|
| 原绝对值数据中，family 均值解释的 `log10σ` 方差比例 | 33.1% |
| 当前趋势记录中，family 均值解释的 `log10σ` 方差比例 | 56.8% |
| 当前趋势记录中，group 均值解释的 `log10σ` 方差比例 | 85.2% |
| 分组前 `log10σ` 总体标准差 | 2.26 |
| 去除 group 基线后的组内标准差 | 0.87 |

这说明按 DOI、温度、制备和 family 配对后，大部分“哪个 family 天生更高”以及实验条件带来的绝对分区差异被条件化掉了。趋势模型更关注组内局部变化，因此原绝对值模型的重要性排名只能作为初始化先验，不能作为趋势模型的最终排序。

高电导率区也高度集中：原绝对值数据中共有 99 条 `σ >= 1e-3 S/cm` 的记录，其中 LGPS 和 argyrodite 合计 60 条。这两个 family 只占全部 430 条记录的约 20.2%，却贡献约 60.6% 的高导记录。原绝对值模型因此可能部分依赖“识别高导 family/分区”；同 family 内配对会显著削弱这条捷径。

两个数据域也不同：当前趋势记录中约 13.4% 的电导率不高于 `1e-6 S/cm`，最低值为 `4.2e-18 S/cm`；原绝对值模型基本在 `>=1e-6 S/cm` 的过滤区间训练。因此直接复用原模型或其特征重要性还会受到分布外数据影响。

`family` 在趋势任务中仍有作用，但作用从“绝对基线”变为“调节变量”：相同的 Li 增减、半径失配或阴离子替换，在 garnet、NASICON 和 halide 中可能产生不同方向和幅度。建议用以下方式验证：

即使先让每个 group 等权，`几乎不变` 的比例仍有明显 family 差异：LGPS 约 1.4%（8 组）、halide 约 19.4%（33 组）、garnet 约 60.2%（21 组）。这支持保留 family 作为交互/调节信息，但不支持把它当作有序数值；小样本 family 的比例不能作强结论。

1. 先训练不含 family 的 delta 模型。
2. 增加规范化 family 类别，比较 grouped-CV 增益。
3. 增加 `family × 关键 delta` 交互，或采用 family-specific 校准。
4. 报告 leave-family-out，区分“已知 family 内插”与“新 family 外推”。

反向 pair 增强还有一个好处：每个有效变化同时出现 `(A,B)` 和 `(B,A)` 后，同一 family 内的“增大/减少”数量天然对称，family 本身便不能利用原始行顺序猜测变化方向；它仍可帮助判断该 family 更容易出现“几乎不变”，以及调节具体 delta 特征的作用。

## 3. 原 35 特征如何迁移到 pair 模型

### 3.1 总体转换规则

| 原特征类别 | 原特征 | pair 转换建议 | 优先级 |
|---|---|---|---|
| 电负性 | `χₐₗₗ`、`χ⁻`、`χ⁺(excl Li⁺)`、两种阳/阴离子差、`χₘₐₓ-χₘᵢₙ` | 所有特征保留 `Δ`；对 `χₐₗₗ`、两种阳/阴离子差增加 `mean` 和 `|Δ|` | 高 |
| 离子/键半径 | `rₐₗₗ`、`r⁻`、两种 `r⁺`、两种半径差、半径比 | 所有特征保留 `Δ`；半径差和半径比增加 `mean`、`|Δ|` 或 log-ratio | 高 |
| 电荷密度 | 三种 `ρ` 和 `ρ⁺/ρ⁻` | 保留 `Δ`；比值优先使用 `Δlog`；绝对电荷密度的 mean 作为宿主背景 | 中高 |
| 离子场强 | 两种 `Φ⁺` | 保留 `Δ`；对含 Li 的场强增加 mean | 中 |
| 化学计量 | `nₕₒₛₜ cₐₜᵢₒₙ`、`n_Li` | 二者在现有代码中已经是总原子分数，必须保留 signed `Δ`；可新增 Li/阴离子或 Li/宿主位比 | 高 |
| family | `family` | 不做差值；规范化类别后作为上下文 | 高，但不用 ordinal |
| 人工交互项 | `r_ratio_x_chi_diff`、`log_rho_ratio`、`n_Li×r⁻`、`chi_range_x_r_avg`、`field_x_r_diff` | 首轮只保留重要交互项的 `Δ`；避免与基础特征重复扩张 | 中 |
| Small/Kong 统计 | 8 个组成统计特征 | 重点保留 signed `Δ`；高排名项增加 mean 和 `|Δ|` | 高 |

### 3.2 历史绝对值模型的重要性（仅作弱先验）

按 Decision Tree、Random Forest、NGBoost 和 LightGBM 的归一化重要性及平均名次综合，历史排序如下。由于原切分存在 DOI 重叠，表中顺位不代表趋势任务优先级：

| 历史顺位 | 原特征 | pair 中的首选表示 | 物理含义 |
|---:|---|---|---|
| 1 | `d_mean_c` | `Δ`、`mean`、`|Δ|` | 阳离子 d 电子背景及变化 |
| 2 | `r_mean_l` | `Δ`、`mean`、`|Δ|` | 非 Li 元素平均键半径 |
| 3 | `mp_variance_c_l` | `Δ`、`mean` | 宿主阳离子熔点离散度 |
| 4 | `χₐₗₗ` | `Δ`、`mean`、`|Δ|` | 全组成平均电负性 |
| 5 | `χ⁺(excl Li⁺)-χ⁻` | `Δ`、`mean`、`|Δ|` | 宿主阳离子与阴离子的离子性差异 |
| 6 | `r⁺(excl Li⁺)-r⁻` | `Δ`、`mean`、`|Δ|` | 宿主阳/阴离子尺寸失配 |
| 7 | `ir_mean_square_all` | `Δ`、`mean` | 离子半径二阶统计量 |
| 8 | `atwt_geometric_mean_all` | `Δ`、`mean` | 整体原子质量尺度 |
| 9 | `entropy_c_l` | `Δ`、`mean` | 宿主阳离子组成熵 |
| 10 | `Vs_variance_c_l` | `Δ`、`mean` | 宿主阳离子 s 轨道体积离散度 |
| 11 | `χ⁺(incl Li⁺)-χ⁻` | `Δ`、`mean` | 含 Li 阳离子与阴离子电负性差 |
| 12 | `field_x_r_diff` | `Δ` | 场强与半径差交互作用 |
| 13 | `ρ⁺(incl Li⁺)/ρ⁻` | `Δlog`、`mean` | 含 Li 阳/阴离子电荷密度匹配 |
| 14 | `chi_range_x_r_avg` | `Δ` | 电负性跨度与平均半径交互作用 |
| 15 | `n_Li` | `Δ` | Li 原子分数和载流子相关组成变化 |

注意：绝对值模型中的重要性不是因果关系，也不保证能迁移到趋势任务。特别是 `χ⁻`、`r⁻` 在绝对值模型中排名较低，但对 O→S→Se 或 Cl→Br→I 这类阴离子替换的 pair 可能非常重要，因此仍应进入 full candidate pool。

### 3.3 条件化诊断后的优先级修正

为直接检查当前趋势数据，已对进入配对的 647 条唯一 `group_id + ID` 记录、569 个唯一化学式重新计算原 34 个数值特征，并比较：

- 绝对值数据中的单变量相关。
- 每个趋势 group 去均值后的组内相关。
- 限制到与原绝对值模型相同的 `σ >= 1e-6 S/cm` 后的相关。
- 每个 group 等权后的相关。
- 2539 个 pair 的 `Δfeature` 与 `Δlog10σ` 相关。

限制到 `σ >= 1e-6 S/cm` 后，趋势侧仍有 553 条记录、105 个 group 和 2235 个 pair。

#### 趋势中明显增强且较稳定的特征

| 特征 | 绝对值 Pearson r | group 去均值 r | 阈值对齐 r | 对齐后 group 等权 r | pair Δ r |
|---|---:|---:|---:|---:|---:|
| `entropy_c_l` | -0.056 | 0.205 | 0.224 | 0.176 | 0.281 |
| `ρ⁺(incl Li⁺)` | -0.013 | -0.143 | -0.226 | -0.243 | -0.225 |
| `Φ⁺(incl Li⁺)` | 0.067 | -0.177 | -0.241 | -0.267 | -0.263 |
| `field_x_r_diff` | -0.136 | 0.130 | 0.241 | 0.253 | 0.218 |

其中 `entropy_c_l` 的方向在全量、阈值对齐、group 等权和 pair Δ 分析中最稳定。含 Li 阳离子的电荷密度、离子势以及 `field_x_r_diff` 在绝对值任务中很弱，但在局部变化任务中明显增强。

上表相关性使用原始数值。实际建模时 `ρ⁺` 和 `Φ⁺` 为正值，建议首选 log-difference 以降低尺度和极端值影响，并把原始 signed delta 留作折内消融对照。

#### 原绝对值相关性明显下降的特征

| 特征 | 绝对值 Pearson r | group 去均值 r | 阈值对齐 r | pair Δ r |
|---|---:|---:|---:|---:|
| `d_mean_c` | -0.351 | -0.029 | -0.146 | 0.028 |
| `χ⁻` | -0.322 | -0.002 | 0.008 | 0.036 |
| `r⁺(excl Li⁺)-r⁻` | -0.351 | 0.056 | 0.102 | 0.129 |
| `χ⁺(incl Li⁺)-χ⁻` | 0.331 | -0.046 | -0.111 | -0.092 |
| `Φ⁺(excl Li⁺)` | 0.337 | -0.082 | -0.186 | -0.177 |

这些结果直接支持“绝对值与趋势的重要性会不同”。部分原高排名特征主要描述 family 之间的全局分区，在同一文献和条件内比较时会显著减弱；另一些含 Li 的局部环境特征则被放大。

相关性只是单变量预筛选，不能代替 grouped-CV 模型重要性。pair 行并不独立，因此优先参考“每条原始记录只出现一次”的 group 去均值和 group 等权结果，pair Δ 只作为方向一致性的辅助验证。

### 3.4 原 35 特征自身的冗余与定义注意事项

- 原 35 特征由 21 个基础组成描述符、1 个 ordinal `family`、5 个确定性交互项和 8 个 Small/Kong 统计特征构成。
- `n_Li` 是 Li 原子数除以总原子数，`nₕₒₛₜ cₐₜᵢₒₙ` 也是宿主阳离子原子分数；不要再把 `delta_Li_fraction` 当成新特征重复计算。
- 完整样本上的 35 维矩阵只有 32 个线性独立维度。三组精确关系来自已同时保留基础量和差值，例如 `χ⁺(excl Li⁺)-χ⁻`、`r⁺(excl Li⁺)-r⁻` 和 `r⁺(incl Li⁺)-r⁻`。
- 5 个人工交互项均由基础列确定，并存在多组高度相关特征。线性模型应从每个精确依赖组中删除一列；树模型也应通过消融避免重要性被重复列稀释。
- `d_mean_c` 和 `s_mean_square_c` 当前统计的是元素全部已占据 d/s 壳层电子数，不等同于价电子数。
- Small 特征遇到缺失元素属性时使用物理数值 `0` 回填。后续最好改为中位数/可解释回填，并增加 missing indicator，避免模型把“数据库缺失”误学成真实零值。

## 4. 推荐的特征版本

### 4.1 V1-compact：建议首先训练

目标是以有限维度验证 pair 表示是否有效。

#### A. 13 个核心 signed delta

- `Δentropy_c_l`
- `Δlog[ρ⁺(incl Li⁺)]`
- `Δlog[Φ⁺(incl Li⁺)]`
- `Δfield_x_r_diff`
- `Δn_Li`
- `Δlog_rho_ratio`
- `Δrₐₗₗ`
- `Δnₕₒₛₜ cₐₜᵢₒₙ`
- `Δir_mean_square_all`
- `Δatwt_geometric_mean_all`
- `Δχₐₗₗ`
- `Δχ⁻`
- `Δr⁻`

前 4 项是当前趋势数据直接支持的首要特征；`Δχ⁻` 和 `Δr⁻` 的全局单变量相关较弱，但用于保留阴离子骨架变化信息。

以下 6 项只作为历史绝对值先验的消融候选，不默认进入最小基线：

- `Δ[χ⁺(incl Li⁺)-χ⁻]`
- `Δd_mean_c`
- `Δr_mean_l`
- `Δmp_variance_c_l`
- `Δ[χ⁺(excl Li⁺)-χ⁻]`
- `Δ[r⁺(excl Li⁺)-r⁻]`

#### B. 8 个宿主背景 mean

- `mean_entropy_c_l`
- `mean_log[ρ⁺(incl Li⁺)]`
- `mean_log[Φ⁺(incl Li⁺)]`
- `mean_field_x_r_diff`
- `mean_n_Li`
- `mean_rₐₗₗ`
- `mean_atwt_geometric_mean_all`
- `mean_χₐₗₗ`

#### C. 6 个变化幅度特征

- `|Δentropy_c_l|`
- `|Δlog[ρ⁺(incl Li⁺)]|`
- `|Δlog[Φ⁺(incl Li⁺)]|`
- `|Δfield_x_r_diff|`
- `|Δn_Li|`
- `composition_L1_distance`

#### D. 组成编辑特征

- `delta_Li_to_anion_ratio`
- `delta_Li_to_host_cation_ratio`
- `added_element_count`
- `removed_element_count`
- `same_element_set`
- `dominant_change_type`：cation、anion、halide、mixed、stoichiometry-only
- `weighted_ionic_radius_mismatch`
- `weighted_electronegativity_mismatch`
- `weighted_oxidation_state_mismatch`
- `weighted_atomic_mass_mismatch`
- `dopant_fraction` 或总变动原子分数

#### E. 类别和状态上下文

- 规范化后的 `family`
- `amorphous_status_a`、`amorphous_status_b`
- `amorphous_transition`：same、crystal→amorphous、amorphous→crystal、unknown
- `crystal_phase_a`、`crystal_phase_b`
- `same_phase`、`phase_known_both`
- 温度的数值表示或 RT 标志
- 规范化后的制备方法类别

类别变量优先交给 CatBoost 原生处理，或使用 one-hot；不要使用没有物理顺序的整数编码。

### 4.2 V1-full：作为候选池，不直接作为最终模型

- 34 个非 family 特征的 signed `Δ`
- 34 个 pair mean
- 重要正值特征的 `Δlog`
- 重要特征的 `|Δ|`
- family、非晶、晶相、温度、制备上下文
- 组成编辑特征

候选特征会超过 80 个，应在每个训练折内部进行特征筛选，不能先在全数据上筛选再交叉验证。

### 4.3 V2：补充结构与工艺

相近工作反复表明，纯组成特征存在明显上限。若原文或数据库中可获得，建议依次增加：

1. 空间群、晶系、晶胞体积、a/b/c、体积/原子。
2. 密度、相对密度、晶粒尺寸和孔隙率。
3. Li 位点占据率、可迁移 Li 比例、配位数和通道维度。
4. 活化能、迁移势垒或其可计算代理量。
5. 烧结/退火温度、时间、压力、球磨、淬火、热压/SPS 等结构化工艺变量。

当前制备方法已用于分组，但仍应把详细工艺文本结构化；“同一种方法”不等于烧结温度、时间和致密度完全相同。

### 4.4 V3：利用原绝对值模型做 stacking

可增加：

- `pred_log_sigma_a`
- `pred_log_sigma_b`
- `pred_delta_log_sigma`
- `pred_uncertainty_a`
- `pred_uncertainty_b`

但这些值必须由严格的 out-of-fold 模型产生。对每个外层测试 DOI，绝对值模型也必须在排除该 DOI 的数据上训练。直接使用在相同材料或相同 DOI 上拟合过的预测值会造成泄漏。

## 5. 35 个统计特征仍缺失的关键信息

原 35 特征把元素属性压缩为均值、方差、比值和交互项，会丢失以下信息：

- 具体新增或移除了哪个元素。
- 哪个元素被哪个元素替代。
- 替位发生在 Li 位、宿主阳离子位还是阴离子位。
- 掺杂量和替位比例。
- 是否只改变 Li 非化学计量，而元素集合不变。
- 同一平均半径变化来自一种大半径掺杂，还是多元素共同调整。

因此，组成编辑特征不是附属项，而是趋势模型与绝对值模型最主要的区别。建议从 A、B 化学式的归一化元素分数差中显式提取：

\[
\Delta x_e=x_{e,B}-x_{e,A}
\]

再根据正、负变化分别得到 added/removed 元素，并计算替换元素之间的电负性、离子半径、氧化态和原子质量差。

若难以可靠识别晶格位点，可先使用 `cation/anion/halide/mixed` 五分类和加权属性失配，避免生成过度确定但错误的位点标签。

## 6. 防止数据泄漏与样本失衡

### 6.1 切分单位

- 主评估：`GroupKFold` 或 `StratifiedGroupKFold`，`group=doi`。
- 不建议仅按 `group_id` 切分，因为一个 DOI 可能对应多个温度或制备组。
- 额外报告 leave-family-out，检查对未见材料家族的泛化。
- 任何标准化、缺失值填充、特征筛选、类别编码和 stacking 都必须只在训练折拟合。

### 6.2 pair 权重

同组 n 条记录会产生 `C(n,2)` 个 pair。建议每个 pair 的训练权重为：

\[
w_{ij}=\frac{1}{\binom{n_g}{2}}
\]

这样每个 group 对损失函数的总贡献接近一致，避免 30 条记录形成的 435 对主导模型。还可以进一步按 DOI 归一化，使每个 DOI 总权重一致。

### 6.3 禁止作为输入的列

以下列只用于审计、分组或构造目标，不能直接作为模型特征：

- `doi`、`group_id`、`group_label`、`pair_id`
- `id_a`、`id_b`、source row
- A/B 的实测电导率
- 电导率绝对变化、倍数、trigger 和所有 trend label

## 7. 模型与目标建议

### 7.1 首轮基线

建议同时训练：

1. 多项逻辑回归：检验特征是否存在稳定线性趋势。
2. LightGBM：与原绝对值工作保持可比。
3. CatBoost：更适合 family、晶相、非晶和制备方法等类别变量。

主指标：

- macro-F1
- balanced accuracy
- MCC
- 三分类混淆矩阵
- 每个 family 的 macro-F1
- swap consistency

不能只报告普通 accuracy，因为 family 和大 group 的组合扩张会掩盖失败模式。

除直接三分类外，建议增加一个与标签规则一致的两阶段模型：

1. `effective_change`：有效变化 vs 几乎不变。
2. 在有效变化样本中预测 direction：增大 vs 减少。

这样 family/宿主 mean 可主要帮助第一阶段判断某类体系是否容易产生有效变化，signed delta 主要负责第二阶段的方向判断。应与直接三分类使用完全相同的 DOI folds 比较，不能只报告较好的一个。

### 7.2 交换一致性

如果交换 A 和 B：

- signed `Δ` 必须变号。
- mean 和 `|Δ|` 必须不变。
- “增大”和“减少”必须互换。
- “几乎不变”必须保持不变。

推荐在训练集中加入反向 pair `(B,A)`，并把标签同步反转；测试时报告模型对正反两个方向是否一致。

### 7.3 连续目标作为辅助任务

可并行预测：

- `Δlog10σ = log10σ_B-log10σ_A`
- `Δσ = σ_B-σ_A`

当前分类规则同时使用 0.1 mS/cm 的绝对差阈值和 100 倍的倍率阈值，仅预测 `Δlog10σ` 不能完整重建标签。更合理的方案是：

- 以直接三分类作为主基线；
- 用 `Δlog10σ` 和 `Δσ` 做多任务辅助；
- 比较“直接分类”与“连续预测后按规则映射”的结果。

## 8. 相近工作汇总

本轮通过 OpenAlex 和 Crossref 检索固态电解质电导率预测、描述符筛选、掺杂优化、工艺文本挖掘和 delta learning。未在本轮结果中发现与“同 DOI、同温度、同制备、同 family 内构造 pair，并预测增大/减少/几乎不变”完全相同的公开工作；多数工作预测绝对电导率或高/低电导率。

| 工作 | 数据/任务与主要特征 | 对本项目的直接启示 |
|---|---|---|
| Hargreaves et al., 2023, [npj Computational Materials](https://doi.org/10.1038/s41524-022-00951-z) | 820 条、214 个来源；403 个近室温唯一组成；数据库含结构标签和温度，但 CrabNet 输入仅为组成；使用 Element Mover's Distance 和 9 个化学簇的 LOCO-CV | 增加 composition distance/OOD 诊断并按 DOI、family/化学簇评估；组成-only 模型不能区分多晶型 |
| Mishra et al., 2023, [ACS Omega](https://doi.org/10.1021/acsomega.3c01400) | 用活化能、工作温度、晶格参数和晶胞体积预测电导率，比较多种 ensemble 模型 | 组成之外，温度、活化能和晶体结构是高价值 V2 特征；若活化能由同一批电导率拟合得到，必须防止目标泄漏 |
| Xu et al., 2020, [JPhys Energy](https://doi.org/10.1088/2399-6528/ab92d8) | 70 个 NASICON 样本；从 47 个元素/晶胞简易描述符筛到 7 个，逻辑回归区分好/差导体，并测试 Na→Li 跨域迁移 | 小数据下应优先 compact 特征；需要检验跨 family 泛化，而非只追求高维模型 |
| Wu et al., 2020, [Science and Technology of Advanced Materials](https://doi.org/10.1080/14686996.2020.1824985) | 区分 bulk 与 grain-boundary conductivity；分析晶粒、晶相、Li 比例、密度、晶胞体积、电负性、极化率、烧结和合成方法 | 工艺、密度和晶粒信息可能解释组成模型无法解释的 pair；Li 比例必须显式加入 |
| Sendek et al., 2018, [Chemistry of Materials](https://doi.org/10.1021/acs.chemmater.8b03272) | ML 引导从 12,000 多个已合成材料中筛选，随后用 DFT-MD 验证；ML 搜索优于随机与人工筛选 | 趋势模型应评价候选排序和 top-k 命中率，不只报告分类准确率 |
| Sendek et al., 2017, [Energy & Environmental Science](https://doi.org/10.1039/C6EE02697D) | 对 12,000 多个候选进行整体筛选，综合结构和多项材料要求 | 最终推荐应同时考虑稳定性、电子绝缘性等约束，不宜仅优化电导率趋势 |
| Zhang et al., 2019, [Nature Communications](https://doi.org/10.1038/s41467-019-13214-1) | 用阴离子子晶格 mXRD 表示结构，并将 2986 个条目归并为 528 个代表结构进行无监督筛选 | 阴离子骨架和结构距离应与组成统计互补；适合构建 anion-framework change 特征 |
| Kang et al., 2023, [The Journal of Physical Chemistry C](https://doi.org/10.1021/acs.jpcc.3c02908) | 从 19,480 个含 Li 材料筛选；使用 chemical descriptor、晶系和原子数，并用 ensemble 与第一性原理验证 | 当前模型应加入晶系、晶相和组成复杂度/原子数特征 |
| Sun et al., 2023, [ACS Applied Materials & Interfaces](https://doi.org/10.1021/acsami.2c15980) | 对 LLZO 的 La/Zr 位进行 73 种元素替位，产生 5329 个候选；使用 surrogate model、active learning 和 AIMD | 替位元素身份、位点、掺杂量应显式编码；后续可用不确定性驱动主动学习 |
| El Massafi et al., 2026, [ACS Applied Materials & Interfaces](https://doi.org/10.1021/acsami.5c18028) | 掺杂 Li2ZrCl6；使用 Element Fraction、Element Property 和原子数，并显式探索二价/三价替位与电荷补偿 | 与当前 pair 最接近，直接支持元素分数差、dopant identity、价态、位点、掺杂量和 charge-balance 特征 |
| Zhao et al., 2025, [Chemistry of Materials](https://doi.org/10.1021/acs.chemmater.5c02633) | Partial Site Occupancies-informed ML；构型熵和层间距是关键描述符，Y3+ 占据效应占主导 | 增加位点占据、无序度、构型熵、层间距及其 A→B 变化 |
| Takeda et al., 2025, [Next Materials](https://doi.org/10.1016/j.nxmate.2025.100574) | 在 Li-rich NASICON 中联合优化 Ca/Si 掺杂量与加热条件；Bayesian optimization 将实验轮次相对穷举减少近 80% | 组成编辑和工艺变量必须联合编码；后续可用不确定性/贝叶斯优化推荐实验 |
| Kong et al., 2025, [Small](https://doi.org/10.1002/smll.202509918) | composition-only E2I 指导 argyrodite 的 Si-Sn、Ge-Si、Ge-Sn 共替位并实验验证；hot pressing 进一步使电导率超过 `1e-2 S/cm` | 共替位身份和比例需要显式特征；同一组成的工艺仍能造成显著变化，组成与工艺必须分开编码 |
| Adhyatma et al., 2022, [Materials Letters](https://doi.org/10.1016/j.matlet.2021.131159) | 面向掺杂 LLZO，以简易描述符优化电导率模型 | 全体系模型之外，应建立 family-specific 或 host-specific 校准模型 |
| Zhang et al., 2024, [Journal of Energy Storage](https://doi.org/10.1016/j.est.2023.109714) | 面向反钙钛矿电解质挖掘离子电导率描述符 | 描述符—趋势关系具有 family 依赖性，应报告分 family 的稳定性 |
| Xiang et al., 2025, [ACS Applied Energy Materials](https://doi.org/10.1021/acsaem.4c02759) | 可解释 ML 分析反钙钛矿；A 位电负性、密度和离子半径是关键特征 | 支持在 pair 中加入明确的位点级电负性、密度和半径变化，并用 SHAP 检查方向 |
| Ma et al., 2024, [Journal of Power Sources](https://doi.org/10.1016/j.jpowsour.2024.234492) | 面向 garnet 体系使用 Gradient Boosting Regression 预测和优化电导率 | 支持把体系专用模型作为全局模型的对照；不能仅凭该工作断言两层校准一定更优 |
| Lee et al., 2024, [The Journal of Physical Chemistry Letters](https://doi.org/10.1021/acs.jpclett.4c00995) | 组合 14 个声子描述符与 16 个结构/电子描述符；声子特征对分类和回归均重要，并用于筛选 264 个含 Li 材料 | 静态组成差分不能完全描述离子输运；声子、晶格动力学和迁移环境应作为后续高价值特征 |
| Mahbub et al., 2020, [Electrochemistry Communications](https://doi.org/10.1016/j.elecom.2020.106860) | 用文本挖掘从大量文献提取硫化物和氧化物 SSE 的合成参数 | 可把备注和制备方法中的烧结、球磨、淬火、温度和时间转成结构化特征 |
| Li et al., 2024, [Advanced Energy Materials](https://doi.org/10.1002/aenm.202304480) | 综述 SSE 的热膨胀、模量、扩散率、电导率、反应能、迁移势垒、带隙和活化能等 ML 任务 | 组成-only 模型只能覆盖部分机制；后续应扩展结构、动力学和多目标性质 |
| Ramakrishnan et al., 2015, [Journal of Chemical Theory and Computation](https://doi.org/10.1021/acs.jctc.5b00099) | 提出 Δ-machine learning：学习相对基线的修正，而不是从零学习绝对量 | 方法论上支持“预测变化量”及利用绝对值模型作为受控基线，但不是固态电解质 pair 任务的直接先例 |

## 9. 推荐实施顺序

1. 复用 `main/features.py` 为 A、B 分别计算原 34 个数值特征，不使用 ordinal family。
2. 构建 V1-compact：signed delta、少量 mean/absolute delta、组成编辑和状态上下文。
3. 规范化 family；当前 pair 数据中仍存在 `halide/halides`、`sulfide/sulfides`、`oxide/oxides`、`argyrodite/argyrodites` 等拆分。
4. 按 DOI 做 grouped cross-validation，并给每个 group/DOI 等权。
5. 训练 Logistic Regression、LightGBM 和 CatBoost 三个基线，加入反向 pair 数据增强。
6. 用 grouped permutation importance 和跨折 SHAP 稳定性决定是否扩展到 V1-full。
7. 第二阶段加入晶胞、密度、晶粒、活化能和结构化工艺特征。
8. 最后再尝试严格 OOF 的绝对值模型 stacking 和 family-specific 校准。

## 10. 最推荐的首轮实验对照

为判断增益来自哪里，建议固定相同 DOI folds，依次比较：

| 实验 | 特征 |
|---|---|
| B0 | 仅 13 个核心 signed delta |
| B1 | B0 + 8 个 pair mean |
| B2 | B1 + composition edit 特征 |
| B3 | B2 + family/晶相/非晶/工艺上下文 |
| B4 | B3 + 反向 pair 增强与 group 权重 |
| B5 | B4 + 严格 OOF 绝对值模型预测 |

如果 B2 相对 B1 有稳定提升，说明“替了谁、替多少”的显式编辑特征是关键；如果 B3 提升明显，则晶相/工艺上下文是主要缺失信息；如果 B5 才显著提升，则原绝对值模型可作为有效先验。
