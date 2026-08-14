"""Create a Chinese infographic for current trend classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/trend/trend_cls_v3_f27_family_abs01_fixedval_optuna10_seed42"
OUT = Path(__file__).resolve().parent
MODELS = ["catboost", "random_forest", "xgboost", "lightgbm"]
NAMES = ["CatBoost", "随机森林", "XGBoost", "LightGBM"]
COLORS = ["#27C2A5", "#5B8FF9", "#F6BD16", "#E8684A"]


def add_card(fig, bounds, facecolor="#121D2E", edgecolor="#263A55"):
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes, linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor,
        clip_on=False,
    )
    ax.add_patch(card)
    return ax


def main():
    font_path = OUT / "NotoSansCJKsc-Regular.otf"
    font_manager.fontManager.addfont(font_path)
    font_family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    validation = pd.read_csv(RUN / "model_comparison.csv").set_index("model").loc[MODELS]
    experimental = pd.read_csv(OUT / "combined_experimental_metrics.csv").set_index("model")

    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="#08111F")
    fig.text(0.055, 0.934, "离子电导率趋势预测｜四种最佳模型对比", color="white", fontsize=25, weight="bold")
    fig.text(0.056, 0.895, "任务：由材料 A→B 的组成变化，预测电导率 增大 / 几乎不变 / 减少", color="#9EB2CC", fontsize=12)
    fig.text(0.945, 0.932, "CURRENT · 2026.08", color="#6D84A1", fontsize=9, ha="right")

    flow = add_card(fig, [0.055, 0.745, 0.89, 0.115])
    flow.text(0.025, 0.70, "模型如何工作", color="#AFC5DD", fontsize=11, weight="bold")
    steps = [("材料对 A → B", "同文献 / 同条件配对"), ("27 个成对特征", "Δ变化 + 均值背景 + 变化幅度"), ("四类树模型", "固定验证集选择与比较"), ("趋势输出", "↑ 增大   ≈ 不变   ↓ 减少")]
    xs = [0.05, 0.31, 0.57, 0.82]
    for i, ((title, sub), x) in enumerate(zip(steps, xs)):
        flow.text(x, 0.40, title, color="white", fontsize=12, weight="bold", ha="center")
        flow.text(x, 0.15, sub, color="#89A0BA", fontsize=8.5, ha="center")
        if i < 3:
            flow.text((x + xs[i + 1]) / 2, 0.35, "→", color="#27C2A5", fontsize=20, ha="center")

    left = add_card(fig, [0.055, 0.335, 0.435, 0.375])
    left.text(0.045, 0.92, "固定验证集表现", color="white", fontsize=15, weight="bold")
    left.text(0.045, 0.855, "同一固定验证集 · 2,120 个配对；数值越高越好", color="#8299B4", fontsize=8.8)
    ax1 = fig.add_axes([0.083, 0.405, 0.38, 0.235], facecolor="none")
    val_metrics = validation[["validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy"]].to_numpy(float) * 100
    y = np.arange(4); h = 0.20
    for j, label in enumerate(["准确率", "宏平均 F1", "平衡准确率"]):
        bars = ax1.barh(y + (j - 1) * h, val_metrics[:, j], h * 0.88, label=label, alpha=[1, .76, .48][j], color=COLORS)
        if j == 0:
            for bar, value in zip(bars, val_metrics[:, j]):
                ax1.text(value + .8, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va="center", color="#D9E5F2", fontsize=8)
    ax1.set_yticks(y, NAMES, color="#CED9E7", fontsize=9); ax1.invert_yaxis(); ax1.set_xlim(0, 70)
    ax1.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60%"], color="#7086A1", fontsize=8)
    ax1.grid(axis="x", color="#23354E", linewidth=.7); ax1.set_axisbelow(True)
    for s in ax1.spines.values(): s.set_visible(False)
    ax1.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor="#B6C7DA")

    right = add_card(fig, [0.51, 0.335, 0.435, 0.375])
    right.text(0.045, 0.92, "外部实验数据表现", color="white", fontsize=15, weight="bold")
    right.text(0.045, 0.855, "合并通用实验 + Halide · 121 对 · 真实标签 68↓ / 9≈ / 44↑", color="#8299B4", fontsize=8.8)
    ax2 = fig.add_axes([0.538, 0.405, 0.38, 0.235], facecolor="none")
    exp_metrics = experimental.loc[MODELS, ["accuracy", "macro_f1", "balanced_accuracy"]].astype(float).to_numpy() * 100
    for j, label in enumerate(["准确率", "宏平均 F1", "平衡准确率"]):
        bars = ax2.barh(y + (j - 1) * h, exp_metrics[:, j], h * .88, label=label, alpha=[1, .76, .48][j], color=COLORS)
        if j == 0:
            for bar, value in zip(bars, exp_metrics[:, j]):
                ax2.text(value + .8, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va="center", color="#D9E5F2", fontsize=8)
    ax2.set_yticks(y, NAMES, color="#CED9E7", fontsize=9); ax2.invert_yaxis(); ax2.set_xlim(0, 70)
    ax2.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60%"], color="#7086A1", fontsize=8)
    ax2.grid(axis="x", color="#23354E", linewidth=.7); ax2.set_axisbelow(True)
    for s in ax2.spines.values(): s.set_visible(False)
    ax2.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor="#B6C7DA")

    bottom = add_card(fig, [0.055, 0.105, 0.89, 0.19])
    bottom.text(0.025, 0.78, "关键结论", color="white", fontsize=14, weight="bold")
    conclusions = [
        ("01  内部验证冠军", "CatBoost：准确率 59.7%，宏平均 F1 60.6%", "#27C2A5"),
        ("02  合并实验集最均衡", "随机森林：准确率 48.8%，宏平均 F1 34.4%，平衡准确率 34.8%", "#5B8FF9"),
        ("03  泛化仍是瓶颈", "合并实验集表现明显低于固定验证集，存在域偏移", "#F6BD16"),
    ]
    for x, (title, sub, color) in zip([0.025, 0.355, 0.685], conclusions):
        bottom.text(x, 0.48, title, color=color, fontsize=11, weight="bold")
        bottom.text(x, 0.25, sub, color="#C0D0E1", fontsize=8.8)
    bottom.text(0.025, 0.06, "注：合并指标按 121 对样本直接汇总，通用实验与 Halide 权重分别为 105/121 和 16/121；阈值 |Δσ| ≤ 1e-4 S/cm。", color="#7188A4", fontsize=7.5)

    output = OUT / "trend_model_comparison_experimental.png"
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
