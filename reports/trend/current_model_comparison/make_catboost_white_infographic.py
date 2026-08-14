"""Create a white-background CatBoost trend prediction infographic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
ORANGE, INK, MUTED = "#F47A20", "#172033", "#64748B"
LINE, PALE = "#DCE3EA", "#FFF4EA"


def card(fig, bounds, face="#FFFFFF", edge=LINE):
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.025",
                               transform=ax.transAxes, linewidth=1.15, edgecolor=edge,
                               facecolor=face, clip_on=False))
    return ax


def metric(ax, x, value, label, note=""):
    ax.text(x, 0.56, value, color=ORANGE, fontsize=25, weight="bold", ha="center")
    ax.text(x, 0.34, label, color=INK, fontsize=10.5, weight="bold", ha="center")
    if note:
        ax.text(x, 0.17, note, color=MUTED, fontsize=7.5, ha="center")


def main():
    font_path = OUT / "NotoSansCJKsc-Regular.otf"
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="#FFFFFF")
    fig.text(0.055, 0.935, "CatBoost 离子电导率趋势预测", color=INK, fontsize=27, weight="bold")
    fig.text(0.055, 0.892, "聚焦一致的卤化物电解质候选体系：绝对值预测 × 趋势预测", color=MUTED, fontsize=13)
    fig.text(0.945, 0.935, "TREND SCREENING · 2026.08", color=ORANGE, fontsize=9, ha="right", weight="bold")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.865, 0.865], transform=fig.transFigure,
                             color=ORANGE, linewidth=3))

    focus = card(fig, [0.055, 0.635, 0.89, 0.19], face=PALE, edge="#F5C69F")
    focus.text(0.025, 0.84, "电导率预测聚焦的 8 个核心候选", color=INK, fontsize=15, weight="bold")
    focus.text(0.025, 0.69, "围绕同一批卤化物电解质，联合判断预测电导率水平与掺杂变化方向。",
               color=MUTED, fontsize=8.8)
    candidates = [
        ("Zr 单掺", "Li2.5Y0.5Zr0.5Cl6", "", ORANGE),
        ("Zr 单掺", "Li2.5Y0.6Zr0.4Cl6", "", ORANGE),
        ("W 单掺", "Li2.7Y0.9W0.1Cl6", "", ORANGE),
        ("Al 单掺", "Li3Y0.9Al0.1Cl6", "", ORANGE),
        ("Al 单掺", "Li3Y0.5Al0.5Cl6", "", MUTED),
        ("Al + Br", "Li3Y0.9Al0.1Cl3.6Br2.4", "", ORANGE),
        ("Zr + Fe", "Li2.9Y0.7Zr0.1Fe0.2Cl6", "", INK),
        ("Zr + Al", "Li2.9Y0.8Zr0.1Al0.1Cl6", "", INK),
    ]
    for i, (system, formula, tag, color) in enumerate(candidates):
        col, row = i % 4, i // 4
        x, y = 0.025 + col * 0.245, 0.47 - row * 0.25
        focus.text(x, y, system, color=color, fontsize=8.2, weight="bold")
        focus.text(x, y - 0.105, formula, color=INK, fontsize=8.5, weight="bold")

    external = card(fig, [0.055, 0.355, 0.43, 0.235])
    external.text(0.045, 0.86, "CatBoost 外部实验数据表现", color=INK, fontsize=15, weight="bold")
    external.text(0.045, 0.73, "通用实验 105 对 + 独立 Halide 16 对 = 121 对", color=MUTED, fontsize=8.8)
    metric(external, 0.31, "47.9%", "准确率", "整体判对比例")
    metric(external, 0.69, "32.1%", "宏平均 F1", "三类等权综合")
    external.text(0.045, 0.055, "真实类别：68 减少 / 9 几乎不变 / 44 增大；阈值 |Δσ| ≤ 1e-4 S/cm",
                  color=MUTED, fontsize=7.6)

    validation = card(fig, [0.515, 0.355, 0.43, 0.235])
    validation.text(0.045, 0.86, "四模型固定验证集对比｜选择 CatBoost", color=INK, fontsize=15, weight="bold")
    validation.text(0.045, 0.73, "424 个固定验证配对；按准确率与宏平均 F1 综合选择", color=MUTED, fontsize=8.8)
    validation.text(0.08, 0.58, "模型", color=MUTED, fontsize=8.5, weight="bold")
    validation.text(0.59, 0.58, "准确率", color=MUTED, fontsize=8.5, weight="bold", ha="center")
    validation.text(0.84, 0.58, "宏平均 F1", color=MUTED, fontsize=8.5, weight="bold", ha="center")
    model_rows = [
        ("CatBoost", "59.7%", "60.6%", True),
        ("随机森林", "58.3%", "57.9%", False),
        ("XGBoost", "55.4%", "54.5%", False),
        ("LightGBM", "52.4%", "53.9%", False),
    ]
    for i, (name, acc, f1, selected) in enumerate(model_rows):
        y = 0.46 - i * 0.105
        color = ORANGE if selected else INK
        validation.text(0.08, y, ("● " if selected else "") + name, color=color,
                        fontsize=8.7, weight="bold" if selected else "normal")
        validation.text(0.59, y, acc, color=color, fontsize=8.7, ha="center",
                        weight="bold" if selected else "normal")
        validation.text(0.84, y, f1, color=color, fontsize=8.7, ha="center",
                        weight="bold" if selected else "normal")
    validation.text(0.045, 0.035, "CatBoost 两项指标均最高 → 作为当前趋势预测模型",
                    color=ORANGE, fontsize=8.1, weight="bold")

    logic = card(fig, [0.055, 0.095, 0.89, 0.215])
    logic.text(0.03, 0.84, "推荐材料的形成逻辑", color=INK, fontsize=15, weight="bold")
    logic.text(0.03, 0.70, "先选出 CatBoost，再将其趋势推荐与绝对值预测进行一致性判定。",
               color=MUTED, fontsize=8.5)
    stages = [
        ("01  绝对值预测", "35 特征回归模型", "估计每个候选的 σ (S/cm)", "筛选高电导率候选并给出峰值位置"),
        ("02  趋势预测", "CatBoost 成对分类", "比较 A→B：增大 / 不变 / 减少", "检查掺杂方向与局部最优位置"),
        ("03  一致性判定", "比较两种推荐结果", "推荐配方或掺杂浓度越接近", "一致性越高 → 实验优先度越高"),
    ]
    xs = [0.035, 0.355, 0.675]
    for i, ((title, model, task, decision), x) in enumerate(zip(stages, xs)):
        logic.text(x, 0.51, title, color=ORANGE, fontsize=10.5, weight="bold")
        logic.text(x, 0.37, model, color=INK, fontsize=9.5, weight="bold")
        logic.text(x, 0.23, task, color=MUTED, fontsize=8.0)
        logic.text(x, 0.09, decision, color=INK, fontsize=7.7)
        if i < 2:
            logic.text(x + 0.285, 0.33, "→", color=ORANGE, fontsize=22, weight="bold", ha="center")

    fig.text(0.055, 0.048, "依据：final_recommendation_confidence.md；外部指标为通用实验与 Halide 按 121 对样本直接汇总。",
             color="#94A3B8", fontsize=7.5)
    output = OUT / "catboost_core_candidates_white.png"
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
