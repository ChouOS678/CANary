"""CANary — CAN 总线异常检测 - 命令行主入口。

This module orchestrates the full pipeline:
1. Load configuration
2. Generate synthetic training data
3. Train RandomForest model
4. Predict on input cases
5. Save outputs (CSV, JSON, charts)

Usage:
    python main.py                  # 标准模式（对照组 float64 列主序）
    python main.py --compare        # 对比模式（对照组 vs 实验组 uint8 直方图）
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from config import validate_domain_config
from data_generator import generate_dynamic_cases, generate_training_rows
from model import (
    format_run_summary,
    make_run_summary,
    predict_cases,
    save_model,
    train_model,
)
from utils import logger
from visualization import (
    render_feature_importance_chart,
    render_stacked_bar_chart,
    render_status_chart,
    render_time_series_chart,
)


def save_training_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Save training data to CSV."""
    from config import FEATURES
    logger.info(f"Saving training CSV to {path}...")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FEATURES, "label"])
        writer.writeheader()
        writer.writerows(rows)


def run_standard_pipeline(group_config: dict, base_dir: Path) -> None:
    """标准模式：对照组 float64 列主序 RandomForest。"""
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    training_rows = generate_training_rows(group_config)

    # Train model
    model, metrics = train_model(training_rows, group_config)

    # Dynamic prediction: generate cases at runtime from blueprints
    input_cases = generate_dynamic_cases(group_config)
    predicted_rows = predict_cases(model, input_cases)

    summary = make_run_summary(
        group_config, training_rows, input_cases, predicted_rows, metrics
    )

    # Save outputs
    save_training_csv(training_rows, output_dir / "synthetic_training_samples.csv")
    save_model(model, output_dir / "model.joblib")

    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predicted_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_summary.txt").write_text(
        format_run_summary(summary), encoding="utf-8"
    )

    # Render charts
    logger.info("Rendering charts...")
    render_status_chart(predicted_rows, output_dir)
    render_time_series_chart(predicted_rows, output_dir)
    render_stacked_bar_chart(predicted_rows, output_dir)
    render_feature_importance_chart(model, output_dir)

    # Print summary
    print(format_run_summary(summary))
    logger.info(f"Output directory: {output_dir}")


def run_compare_pipeline(group_config: dict, base_dir: Path) -> None:
    """对比模式：运行两种算法（传统 scikit-learn vs 直方图离散化），生成效能对比报告。"""
    from perf_compare import PerfBenchmark, run_comparison

    logger.info("运行算法效能对比模式（传统 scikit-learn vs 直方图离散化）...")
    output_dir = base_dir / "output" / "perf_compare"
    results = run_comparison(group_config, output_dir)

    # 保存对比结果 JSON（排除 benchmark 原始数据，减少文件体积）
    (output_dir / "comparison_results.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in results.items()
                if k not in ("benchmark", "controlled_variables")
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    # 保存算法效能分析报告（含根因分析与结论）
    summary_text = PerfBenchmark.summary_text(
        results["benchmark"],
        results["controlled_variables"],
        results["memory_analysis"],
    )
    conclusion_text = PerfBenchmark.generate_conclusion(results)
    (output_dir / "perf_analysis.txt").write_text(
        summary_text + "\n\n" + conclusion_text, encoding="utf-8"
    )

    logger.info(f"算法效能对比报告已保存到 {output_dir}")


def main() -> None:
    """Run the full pipeline. Supports --compare flag for A/B testing."""
    base_dir = Path(__file__).resolve().parent

    compare_mode = "--compare" in sys.argv

    logger.info("Loading configuration...")
    try:
        group_config = json.loads(
            (base_dir / "group_config.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration: {e}")
        raise

    domain_issues = validate_domain_config()
    if domain_issues:
        logger.error("Domain configuration validation failed:")
        for issue in domain_issues:
            logger.error("- %s", issue)
        raise ValueError("Domain configuration is inconsistent; aborting run.")
    logger.info("Domain configuration validation passed.")

    if compare_mode:
        run_compare_pipeline(group_config, base_dir)
    else:
        run_standard_pipeline(group_config, base_dir)


if __name__ == "__main__":
    main()
