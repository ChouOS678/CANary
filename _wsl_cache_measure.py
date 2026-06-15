"""VTune PMC 采集目标脚本 — 供 _vtune_collect.bat 调用。

该脚本仅用于 Intel VTune Profiler 硬件 PMC 事件采集，
在指定算法和样本规模下执行一次完整的训练流程，
使 VTune 能够采集 MEM_LOAD_RETIRED 等微架构事件。

用法（通常由 _vtune_collect.bat 自动调用）：
    python _wsl_cache_measure.py --algo sklearn    --samples 5000
    python _wsl_cache_measure.py --algo histogram  --samples 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import FEATURES, LABELS
from data_generator import generate_training_rows


def _build_group_config(n_samples_per_label: int) -> dict:
    """构造最小化 group_config，供训练函数使用。"""
    config_path = BASE_DIR / "group_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["samples_per_label"] = n_samples_per_label
        return cfg
    # 回退：使用硬编码默认值
    return {
        "samples_per_label": n_samples_per_label,
        "rf_params": {
            "n_estimators": 100,
            "max_depth": 8,
            "random_state": 42,
            "n_jobs": 1,  # VTune 采集时使用单线程以获得清晰 PMC 数据
        },
        "data_profile": {
            "noise_scale": 1.2,
            "boundary_ratio": 0.14,
            "drift_ratio": 0.1,
            "label_noise_ratio": 0.02,
        },
    }


def run_sklearn(group_config: dict) -> None:
    """传统 scikit-learn float64 RandomForest 训练。"""
    from model import train_model

    rows = generate_training_rows(group_config)
    print(f"[sklearn] 训练样本: {len(rows)}")
    train_model(rows, group_config)
    print("[sklearn] 训练完成")


def run_histogram(group_config: dict) -> None:
    """直方图离散化 uint8 RandomForest 训练。"""
    from histogram_model import train_histogram_model

    rows = generate_training_rows(group_config)
    print(f"[histogram] 训练样本: {len(rows)}")
    train_histogram_model(rows, group_config)
    print("[histogram] 训练完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="VTune cache measurement target")
    parser.add_argument(
        "--algo",
        choices=["sklearn", "histogram"],
        required=True,
        help="选择训练算法：sklearn (float64) 或 histogram (uint8)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="每标签样本数（默认 5000）",
    )
    args = parser.parse_args()

    group_config = _build_group_config(args.samples)
    # VTune 采集时禁用并行，确保 PMC 事件归属清晰
    group_config.setdefault("rf_params", {})["n_jobs"] = 1

    print(f"算法: {args.algo} | 每标签样本数: {args.samples} | "
          f"总样本量: {args.samples * len(LABELS)}")

    if args.algo == "sklearn":
        run_sklearn(group_config)
    else:
        run_histogram(group_config)

    print("采集目标执行完毕")


if __name__ == "__main__":
    main()
