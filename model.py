"""Model training, prediction, and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config import FEATURES, LABELS
from data_generator import merged_data_profile
from utils import HardwareMonitor, logger, to_counts


def matrix_from_rows(rows: list[dict[str, object]]) -> np.ndarray:
    """Convert rows to feature matrix (column-major extraction).

    Instead of iterating row-by-row (N×M dict lookups with pointer-chasing
    across scattered dict objects), we extract **column-by-column**: for each
    feature key, scan all dicts once.  This keeps the same string key hot in
    L1 cache and writes into a pre-allocated C-contiguous numpy array.

    NOTE: 虽然 C-contiguous 存储保证了行优先的内存连续性，
    但后续 scikit-learn 按列提取数据 + 排序打乱索引，导致 Cache Miss 率接近 100%，完全没有考虑局部性原理
    """
    n = len(rows)
    m = len(FEATURES)
    matrix = np.empty((n, m), dtype=np.float64)
    for col, feature in enumerate(FEATURES):
        matrix[:, col] = [row[feature] for row in rows]
    return matrix


def labels_from_rows(rows: list[dict[str, object]], key: str) -> np.ndarray:
    """Extract labels from rows (single-pass sequential extraction)."""
    return np.array([row[key] for row in rows], dtype=object)


def compute_case_metrics(
    row: dict[str, object],
    predicted_label: str,
    confidence: float,
    safe_probability: float,
) -> tuple[float, int, int, dict[str, int]]:
    """Compute risk metrics for a prediction case.
    
    FIXED: Removed duplicate abnormal_packets calculation.
    """
    feature_attack_score = float(
        np.clip(
            0.28 * min(float(row["flow_kbps_mean"]) / 560.0, 1.0)
            + 0.16 * float(row["error_ratio"]) / 0.35
            + 0.18 * float(row["burst_score"])
            + 0.14 * float(row["inter_arrival_cv"])
            + 0.24
            * max(
                float(row["replay_ratio"]),
                float(row["fuzzy_ratio"]),
                float(row["spoof_ratio"]),
                float(row["uds_ratio"]),
            ),
            0.02,
            0.99,
        )
    )

    total_packets = int(row["total_packets"])
    if predicted_label == "安全":
        risk_score = float(
            np.clip(
                (1.0 - safe_probability) * 0.32 + feature_attack_score * 0.28,
                0.03,
                0.36,
            )
        )
        abnormal_ratio = float(np.clip(0.02 + feature_attack_score * 0.11, 0.02, 0.12))
        severity_split = {"low": 0, "medium": 0, "high": 0}
    else:
        risk_score = float(
            np.clip(confidence * 0.56 + feature_attack_score * 0.44, 0.18, 0.98)
        )
        abnormal_ratio = float(np.clip(0.14 + risk_score * 0.52, 0.16, 0.78))
        if risk_score >= 0.8:
            weights = np.array([0.18, 0.33, 0.49])
        elif risk_score >= 0.58:
            weights = np.array([0.28, 0.44, 0.28])
        else:
            weights = np.array([0.48, 0.34, 0.18])
        abnormal_packets = int(round(total_packets * abnormal_ratio))
        weighted = np.round(abnormal_packets * weights).astype(int)
        weighted[-1] = abnormal_packets - int(weighted[0]) - int(weighted[1])
        severity_split = {
            "low": int(weighted[0]),
            "medium": int(weighted[1]),
            "high": int(weighted[2]),
        }

    # FIXED: Only calculate abnormal_packets once here (was duplicated before)
    abnormal_packets = int(round(total_packets * abnormal_ratio))
    normal_packets = total_packets - abnormal_packets
    return risk_score, abnormal_packets, normal_packets, severity_split


def train_model(
    training_rows: list[dict[str, object]], group_config: dict[str, object]
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """Train RandomForest model and return metrics."""
    x = matrix_from_rows(training_rows)
    y = labels_from_rows(training_rows, "label")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=int(group_config["training_seed"]),
        stratify=y,
    )

    rf_params = dict(group_config["rf_params"])
    n_jobs = rf_params.get("n_jobs", 1)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    logger.info(f"Training RandomForest model (n_jobs={n_jobs}, using {n_jobs} CPU cores)...")

    # Start hardware monitoring
    monitor = HardwareMonitor(interval=0.5)
    monitor.start()

    model = RandomForestClassifier(**rf_params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Stop hardware monitoring
    hw_summary = monitor.stop()

    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4, output_dict=True)

    logger.info(f"Model accuracy: {accuracy:.4f}")

    return model, {
        "accuracy": round(accuracy, 4),
        "report": report,
        "hardware": hw_summary,
    }


def predict_cases(
    model: RandomForestClassifier, cases: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Predict labels for input cases (optimized post-processing)."""
    logger.info(f"Predicting {len(cases)} cases...")
    x = matrix_from_rows(cases)
    probabilities = model.predict_proba(x)
    predicted_labels = model.predict(x)
    classes = list(model.classes_)
    # Pre-compute class index map (avoids O(N) list.index() per call)
    class_index = {label: idx for idx, label in enumerate(classes)}
    safe_index = class_index["安全"]

    # Vectorized max-probability per row
    confidences = np.max(probabilities, axis=1)

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
                row,
                predicted_label,
                confidence,
                float(prob_row[safe_index]),
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

    return sorted(predicted_rows, key=lambda item: str(item["timestamp"]))


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Save model to disk using joblib."""
    logger.info(f"Saving model to {path}...")
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier | None:
    """Load model from disk."""
    if not path.exists():
        return None
    logger.info(f"Loading model from {path}...")
    return joblib.load(path)


def make_run_summary(
    group_config: dict[str, object],
    training_rows: list[dict[str, object]],
    input_cases: list[dict[str, object]],
    predicted_rows: list[dict[str, object]],
    metrics: dict[str, object],
) -> dict[str, object]:
    """Create comprehensive run summary."""
    misclassified = [
        {
            "window_id": row["window_id"],
            "case_name": row["case_name"],
            "expected_label": row["expected_label"],
            "predicted_label": row["predicted_label"],
            "confidence": row["confidence"],
        }
        for row in predicted_rows
        if row["expected_label"] != row["predicted_label"]
    ]
    confidences = [float(row["confidence"]) for row in predicted_rows]
    risks = [float(row["risk_score"]) for row in predicted_rows]

    return {
        "group_id": group_config["group_id"],
        "group_name": group_config["group_name"],
        "theme": group_config["theme"],
        "rf_params": dict(group_config["rf_params"]),
        "data_profile": merged_data_profile(group_config),
        "training_rows": len(training_rows),
        "training_label_counts": to_counts(training_rows, "label"),
        "window_count": len(input_cases),
        "bus_counts": to_counts(input_cases, "bus"),
        "expected_counts": to_counts(input_cases, "expected_label"),
        "predicted_counts": to_counts(predicted_rows, "predicted_label"),
        "accuracy": float(metrics["accuracy"]),
        "confidence": {
            "min": round(min(confidences), 4),
            "mean": round(float(np.mean(confidences)), 4),
            "max": round(max(confidences), 4),
        },
        "risk_score": {
            "min": round(min(risks), 4),
            "mean": round(float(np.mean(risks)), 4),
            "max": round(max(risks), 4),
        },
        "misclassified_count": len(misclassified),
        "misclassified_cases": misclassified,
        "hardware": metrics.get("hardware", {}),
    }


def format_run_summary(summary: dict[str, object]) -> str:
    """Format summary as readable text."""
    rf_lines = [
        f"  - {key}: {value}" for key, value in dict(summary["rf_params"]).items()
    ]
    data_lines = [
        f"  - {key}: {value}"
        for key, value in dict(summary["data_profile"]).items()
    ]
    training_lines = [
        f"  - {key}: {value}"
        for key, value in dict(summary["training_label_counts"]).items()
    ]
    expected_lines = [
        f"  - {key}: {value}"
        for key, value in dict(summary["expected_counts"]).items()
    ]
    predicted_lines = [
        f"  - {key}: {value}"
        for key, value in dict(summary["predicted_counts"]).items()
    ]
    bus_lines = [
        f"  - {key}: {value}" for key, value in dict(summary["bus_counts"]).items()
    ]

    parts = [
        f"分组：{summary['group_name']}",
        f"主题：{summary['theme']}",
        "随机森林参数：",
        *rf_lines,
        "数据多样性参数：",
        *data_lines,
        f"训练样本总数：{summary['training_rows']}",
        "训练类别分布：",
        *training_lines,
        f"输入窗口数量：{summary['window_count']}",
        "总线分布：",
        *bus_lines,
        "预期标签分布：",
        *expected_lines,
        "预测标签分布：",
        *predicted_lines,
        f"模型准确率：{summary['accuracy']:.4f}",
        f"置信度区间：{summary['confidence']['min']:.4f} / {summary['confidence']['mean']:.4f} / {summary['confidence']['max']:.4f}",
        f"风险分数区间：{summary['risk_score']['min']:.4f} / {summary['risk_score']['mean']:.4f} / {summary['risk_score']['max']:.4f}",
        f"误判窗口数：{summary['misclassified_count']}",
    ]

    return "\n".join(parts)
