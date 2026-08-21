"""Unified command-line prediction entry point for absolute and trend models.

Examples::

    python main/predict.py absolute --model-run runs/absolute/abs_v2_f37_native_family_lgbm_trials50_seed42 --formula Li7La3Zr2O12
    python main/predict.py trend --model runs/trend/.../catboost/model.joblib --raw-csv data/experimental/raw/experimental-data.csv --annotations data/experimental/annotations/experimental-data-labeled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Permit ``python main/predict.py`` from the repository root.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.absolute.predict import PredictConfig, predict_formulas
from main.trend.pipeline import TrendPredictConfig, default_trend_predict_config
from main.trend.predict import predict_trend


def _model_file(value: str, model_name: str | None) -> Path:
    path = Path(value)
    if path.is_dir():
        if model_name:
            name = model_name
        elif (path / "best_model.txt").exists():
            name = (path / "best_model.txt").read_text(encoding="utf-8").strip()
        else:
            name = "lightgbm"
        path = path / name / "model.joblib"
    return path


def _absolute(args: argparse.Namespace) -> None:
    formulas = args.formula or []
    if not formulas and not args.input:
        raise SystemExit("absolute 任务需要 --formula（可重复）或 --input")
    if formulas and args.input:
        raise SystemExit("--formula 与 --input 不能同时使用")
    if args.family and not formulas:
        raise SystemExit("--family 仅与 --formula 配合使用；CSV 请提供 Family 列")

    source: str | pd.DataFrame
    if formulas:
        families = args.family or []
        if len(families) == 1 and len(formulas) > 1:
            families = families * len(formulas)
        if families and len(families) != len(formulas):
            raise SystemExit(
                "--family 数量必须为 1（应用于全部配方）或与 --formula 数量一致"
            )
        source = pd.DataFrame({"True Composition": formulas})
        if families:
            source["Family"] = families
    else:
        source = args.input

    try:
        result = predict_formulas(
            source,
            args.model_run,
            PredictConfig(
                model_name=args.model_name,
                dataset_name=args.dataset,
                output_dir=args.output,
                formula_column=args.formula_column,
                id_column=args.id_column,
                family_column=args.family_column,
                prediction_purpose=args.prediction_purpose,
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(result.predictions.to_string(index=False))
    print(f"输出目录: {result.output_dir.resolve()}")


def _trend(args: argparse.Namespace) -> None:
    defaults = default_trend_predict_config()
    model = _model_file(args.model, args.model_name)
    config = TrendPredictConfig(
        raw_csv=Path(args.raw_csv or defaults.raw_csv),
        annotations=Path(args.annotations or defaults.annotations),
        model=model,
        output=Path(args.output or defaults.output),
        metrics=Path(args.metrics or defaults.metrics),
        raw_scale=args.raw_scale,
        threshold_s_cm=args.threshold,
    )
    metrics = predict_trend(config)
    print(f"预测对数: {metrics['pairs']}")
    print(f"输出文件: {Path(config.output).resolve()}")
    print(f"指标文件: {Path(config.metrics).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="指定模型进行绝对电导率或趋势预测")
    sub = parser.add_subparsers(dest="task", required=True)
    absolute = sub.add_parser("absolute", help="绝对电导率回归")
    absolute.add_argument("--model-run", required=True, help="训练运行目录")
    absolute.add_argument("--model-name", help="模型子目录，如 lightgbm")
    absolute.add_argument("--formula", action="append", help="化学式，可重复")
    absolute.add_argument(
        "--family",
        action="append",
        help="与 --formula 对应的材料 family；单值可应用于全部配方",
    )
    absolute.add_argument("--input", help="输入 CSV/TSV/每行一个化学式")
    absolute.add_argument("--formula-column")
    absolute.add_argument("--family-column", help="批量输入中的 family 列名")
    absolute.add_argument("--id-column", help="批量输入中的样本 ID 列名")
    absolute.add_argument("--prediction-purpose", help="写入预测元数据的用途说明")
    absolute.add_argument("--dataset", default="custom")
    absolute.add_argument("--output", type=Path)
    absolute.set_defaults(func=_absolute)

    trend = sub.add_parser("trend", help="趋势分类")
    trend.add_argument("--model", required=True, help="model.joblib 或训练运行目录")
    trend.add_argument("--model-name", help="运行目录下的模型名")
    trend.add_argument("--raw-csv")
    trend.add_argument("--annotations")
    trend.add_argument("--output")
    trend.add_argument("--metrics")
    trend.add_argument("--raw-scale", type=float, default=1.0)
    trend.add_argument("--threshold", type=float, default=1e-7)
    trend.set_defaults(func=_trend)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
