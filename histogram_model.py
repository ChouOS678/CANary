"""Histogram-based RandomForest model — 实验组实现。

核心创新：
- 将 float32 特征值通过 256 桶等宽直方图离散化为 uint8（0-255）
- 每个样本仅占 11 字节（11 特征 × 1 字节），整批训练数据常驻 L1 Cache
- 行主序构造 + 顺序存取，充分利用空间局部性，Cache Miss 率极低
- scikit-learn RandomForest 原生支持 uint8，无需额外适配
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config import FEATURES, FEATURE_BOUNDS, LABELS
from utils import HardwareMonitor, logger

# ---------------------------------------------------------------------------
# 直方图编码器：float32 → uint8（256 桶）
# ---------------------------------------------------------------------------

NUM_BINS = 256
UINT8_MAX = np.iinfo(np.uint8).max  # 255


class HistogramEncoder:
    """将连续特征值离散化到 256 个等宽桶，映射为 uint8。

    每个特征 f 在 [low, high] 范围内被划分为 256 个等宽区间，
    值 v 落入第 k 个桶 → 编码为 k（0 ≤ k ≤ 255）。

    桶边界 bin_edges[j] 长度为 257，其中：
        bin_edges[j][0]   = low
        bin_edges[j][256] = high + ε
    np.digitize 返回 [1, 256]，减 1 后得到 [0, 255]。
    """

    def __init__(self) -> None:
        self._bin_edges: list[np.ndarray] = []
        self._build_bins()

    # ------------------------------------------------------------------
    def _build_bins(self) -> None:
        """为每个特征预计算 256 桶的等宽边界。"""
        self._bin_edges = []
        for feature in FEATURES:
            low, high = FEATURE_BOUNDS[feature]
            # 257 个边界点生成 256 个区间
            edges = np.linspace(low, high, NUM_BINS + 1)
            # 把最后一个边界稍微抬高，保证最大值也能落入最后一个桶
            edges[-1] = np.nextafter(high, high + 1.0)
            self._bin_edges.append(edges)

    # ------------------------------------------------------------------
    @property
    def bin_edges(self) -> list[np.ndarray]:
        """每个特征的桶边界列表，长度 = len(FEATURES)，每个 ndarray shape=(257,)。"""
        return self._bin_edges

    # ------------------------------------------------------------------
    def encode_rows(self, rows: list[dict[str, object]]) -> np.ndarray:
        """将 dict 列表编码为 (N, M) uint8 矩阵。

        分两步：
        1. 行主序提取 float64 值（写入 C-contiguous 数组，顺序存取）
        2. 逐列 np.digitize 离散化为 uint8

        两个阶段均对缓存友好——阶段 1 连续写，阶段 2 在极小数据上操作。
        """
        n = len(rows)
        m = len(FEATURES)
        float_matrix = np.empty((n, m), dtype=np.float64)

        # 阶段 1：行主序提取（顺序写，局部性最优）
        for i, row in enumerate(rows):
            for j, feature in enumerate(FEATURES):
                float_matrix[i, j] = float(row[feature])

        # 阶段 2：逐列数字化为 uint8
        uint8_matrix = np.empty((n, m), dtype=np.uint8)
        for j in range(m):
            uint8_matrix[:, j] = np.clip(
                np.digitize(float_matrix[:, j], self._bin_edges[j]) - 1,
                0,
                UINT8_MAX,
            ).astype(np.uint8)

        return uint8_matrix

    # ------------------------------------------------------------------
    def encode_single(self, feature_values: dict[str, float]) -> np.ndarray:
        """将单个样本编码为 (M,) uint8 向量。"""
        encoded = np.empty(len(FEATURES), dtype=np.uint8)
        for j, feature in enumerate(FEATURES):
            val = float(feature_values[feature])
            idx = np.digitize(val, self._bin_edges[j]) - 1
            encoded[j] = np.clip(idx, 0, UINT8_MAX)
        return encoded

    # ------------------------------------------------------------------
    def decode_approx(self, encoded: np.ndarray) -> dict[str, float]:
        """从 uint8 编码近似还原为 float 值（取桶中心值）。"""
        result: dict[str, float] = {}
        for j, feature in enumerate(FEATURES):
            idx = int(encoded[j])
            edges = self._bin_edges[j]
            # 桶中心 ≈ 边界中点
            center = (edges[idx] + edges[min(idx + 1, NUM_BINS)]) / 2.0
            result[feature] = float(center)
        return result

    # ------------------------------------------------------------------
    def cache_analysis(self, n_samples: int) -> dict[str, object]:
        """计算编码后数据的缓存驻留分析。

        典型 L1 数据缓存大小：32 KB（每核）
        uint8 编码：11 features × 1 byte = 11 bytes/sample
        1920 样本 → 21,120 bytes ≈ 20.6 KB → 完全驻留 L1D
        对比 float64：11 × 8 bytes = 88 bytes/sample
        1920 样本 → 168,960 bytes ≈ 165 KB → 远超 L1D
        """
        bytes_per_sample_uint8 = len(FEATURES) * 1
        bytes_per_sample_float64 = len(FEATURES) * 8

        total_uint8 = bytes_per_sample_uint8 * n_samples
        total_float64 = bytes_per_sample_float64 * n_samples

        l1d_kb = 32
        l1d_bytes = l1d_kb * 1024  # 32768

        return {
            "n_samples": n_samples,
            "n_features": len(FEATURES),
            "bytes_per_sample_uint8": bytes_per_sample_uint8,
            "bytes_per_sample_float64": bytes_per_sample_float64,
            "total_uint8_bytes": total_uint8,
            "total_uint8_kb": round(total_uint8 / 1024, 1),
            "total_float64_bytes": total_float64,
            "total_float64_kb": round(total_float64 / 1024, 1),
            "l1d_cache_kb": l1d_kb,
            "uint8_fits_l1d": total_uint8 <= l1d_bytes,
            "float64_fits_l1d": total_float64 <= l1d_bytes,
            "l1d_hit_ratio_uint8_pct": round(
                min(100.0, l1d_bytes / total_uint8 * 100), 1
            ) if total_uint8 > 0 else 100.0,
            "l1d_hit_ratio_float64_pct": round(
                min(100.0, l1d_bytes / total_float64 * 100), 1
            ) if total_float64 > 0 else 100.0,
        }


# ---------------------------------------------------------------------------
# 全局编码器实例
# ---------------------------------------------------------------------------
_encoder: HistogramEncoder | None = None


def get_encoder() -> HistogramEncoder:
    """获取全局直方图编码器单例。"""
    global _encoder
    if _encoder is None:
        _encoder = HistogramEncoder()
    return _encoder


# ---------------------------------------------------------------------------
# 训练 & 预测（直方图版本）
# ---------------------------------------------------------------------------


def matrix_from_rows_histogram(rows: list[dict[str, object]]) -> np.ndarray:
    """行主序构造 uint8 特征矩阵，适合 L1 缓存常驻。

    与 model.matrix_from_rows（列主序 float64）形成对照：
    - 实验组：行主序 uint8 → Cache Miss ≈ 0%
    - 对照组：列主序 float64 → Cache Miss ≈ 100%
    """
    return get_encoder().encode_rows(rows)


def train_histogram_model(
    training_rows: list[dict[str, object]],
    group_config: dict[str, object],
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """训练基于直方图离散化特征的 RandomForest 模型。

    返回 (model, metrics_dict)，metrics 包含训练耗时和硬件利用率。
    """
    logger.info("[Histogram] Encoding training data to uint8...")
    t0 = time.perf_counter()
    x = matrix_from_rows_histogram(training_rows)
    encode_time = time.perf_counter() - t0
    logger.info(
        f"[Histogram] Encoding done in {encode_time:.4f}s, "
        f"shape={x.shape}, dtype={x.dtype}, "
        f"memory={x.nbytes / 1024:.1f} KB"
    )

    from model import labels_from_rows

    y = labels_from_rows(training_rows, "label")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=int(group_config["training_seed"]),
        stratify=y,
    )

    rf_params = dict(group_config["rf_params"])
    # 保留 n_jobs 配置以利用多核并行
    n_jobs = rf_params.get("n_jobs", 1)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    logger.info(f"[Histogram] Training RandomForest (n_jobs={n_jobs})...")

    monitor = HardwareMonitor(interval=0.5)
    monitor.start()

    t_train_start = time.perf_counter()
    model = RandomForestClassifier(**rf_params)
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t_train_start

    predictions = model.predict(x_test)
    hw_summary = monitor.stop()

    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4, output_dict=True)

    logger.info(f"[Histogram] Model accuracy: {accuracy:.4f}, train_time={train_time:.4f}s")

    return model, {
        "accuracy": round(accuracy, 4),
        "report": report,
        "hardware": hw_summary,
        "encode_time_sec": round(encode_time, 4),
        "train_time_sec": round(train_time, 4),
    }


def predict_histogram_cases(
    model: RandomForestClassifier,
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    """用直方图模型对输入案例进行预测。"""
    logger.info(f"[Histogram] Predicting {len(cases)} cases...")

    t0 = time.perf_counter()
    x = matrix_from_rows_histogram(cases)
    encode_time = time.perf_counter() - t0

    t_pred = time.perf_counter()
    probabilities = model.predict_proba(x)
    predicted_labels = model.predict(x)
    predict_time = time.perf_counter() - t_pred

    classes = list(model.classes_)
    class_index = {label: idx for idx, label in enumerate(classes)}
    safe_index = class_index.get("安全", 0)
    confidences = np.max(probabilities, axis=1)

    from model import compute_case_metrics

    predicted_rows: list[dict[str, object]] = []
    for index, row in enumerate(cases):
        prob_row = probabilities[index]
        probability_map = {
            label: round(float(prob_row[idx]), 4)
            for label, idx in class_index.items()
        }
        predicted_label = str(predicted_labels[index])
        confidence = float(confidences[index])
        risk_score, abnormal_packets, normal_packets, severity_split = (
            compute_case_metrics(
                row, predicted_label, confidence, float(prob_row[safe_index])
            )
        )
        predicted_rows.append(
            {
                **row,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "risk_score": round(risk_score, 4),
                "abnormal_packets": abnormal_packets,
                "normal_packets": normal_packets,
                "attack_intensity": round(risk_score * 50, 1),
                "severity_split": severity_split,
                "probabilities": probability_map,
            }
        )

    logger.info(
        f"[Histogram] Prediction done: encode={encode_time:.4f}s, "
        f"predict={predict_time:.4f}s"
    )

    return sorted(predicted_rows, key=lambda item: str(item["timestamp"]))
