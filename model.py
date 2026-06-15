"""Model training, prediction, and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import DATA_PROFILE_CN, FEATURES, LABELS, RF_PARAMS_CN
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
    missing: list[str] = []
    for col, feature in enumerate(FEATURES):
        try:
            matrix[:, col] = [row[feature] for row in rows]
        except KeyError:
            missing.append(feature)
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
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

    Theoretical Basis of Weight Coefficients
    ----------------------------------------
    The feature_attack_score is a linear combination of 5 groups of indicators,
    each normalized to [0, 1].  The weights sum to 1.0 and reflect:

    1. flow_kbps_mean (w=0.28) — 流量速率指标
       DoS 攻击的首要特征是流量激增。除以 560 kbps（FEATURE_BOUNDS 上界）归一化。
       权重最高，因为流量异常是 CAN 总线攻击最直接的物理层信号。

    2. error_ratio (w=0.16) — 错误率指标
       模糊攻击和 DoS 均会导致总线错误率上升。除以 0.35（上界）归一化。
       权重适中，因为错误也可能来自正常的总线竞争。

    3. burst_score (w=0.18) — 突发性指标
       攻击报文通常以突发方式注入，与正常周期性 CAN 报文形成对比。
       该指标直接参与 RandomForest 的 burst_score 特征。

    4. inter_arrival_cv (w=0.14) — 时间间隔变异系数
       攻击注入会破坏 CAN 报文的周期性规律，导致到达间隔变异增大。
       权重较低，因为网络抖动也可能引起类似变化。

    5. max(replay, fuzzy, spoof, uds) (w=0.24) — 主导攻击类型比率
       四种攻击类型（重放/模糊/欺骗/UDS）共享此权重。使用 max() 而非 sum()
       是因为在单一时间窗口内，通常只有一种攻击类型占主导（攻击不会同时发生），
       取最大值可避免对多类型叠加场景的过度敏感。
       DOMINANT_FEATURE 映射表定义了每种攻击类型对应的核心特征。

    Risk Score Composition
    ---------------------
    - 预测为「安全」时: 1-safe_probability (0.32) + feature_attack_score (0.28)
      此时模型置信度权重更高，因为"安全"判定更依赖模型判断。
    - 预测为攻击时: confidence (0.56) + feature_attack_score (0.44)
      攻击判定下，模型置信度贡献 56%，特征层信号贡献 44%。
      两者权重比 ≈ 1.27:1，反映模型判断的主导性。

    Severity Split (only for attack predictions)
    --------------------------------------------
    根据 risk_score 分三档，按高一中一低的比例分配异常包数：
      - risk ≥ 0.8: 高:中:低 = 49:33:18（高风险场景下大部分包被标记为高危）
      - risk ≥ 0.58: 高:中:低 = 28:44:28（中等风险，中等占比最高呈正态分布）
      - risk < 0.58: 高:中:低 = 18:34:48（低风险，多数为低等级）
    该分段策略参考了 NIST SP 800-61 的事件严重等级分级思想。
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
        abnormal_packets = int(round(total_packets * abnormal_ratio))
        severity_split = {"low": 0, "medium": 0, "high": 0}
    else:
        risk_score = float(
            np.clip(confidence * 0.56 + feature_attack_score * 0.44, 0.18, 0.98)
        )
        abnormal_ratio = float(np.clip(0.14 + risk_score * 0.52, 0.16, 0.78))
        abnormal_packets = int(round(total_packets * abnormal_ratio))
        if risk_score >= 0.8:
            weights = np.array([0.18, 0.33, 0.49])
        elif risk_score >= 0.58:
            weights = np.array([0.28, 0.44, 0.28])
        else:
            weights = np.array([0.48, 0.34, 0.18])
        weighted = np.round(abnormal_packets * weights).astype(int)
        weighted[-1] = abnormal_packets - int(weighted[0]) - int(weighted[1])
        severity_split = {
            "low": int(weighted[0]),
            "medium": int(weighted[1]),
            "high": int(weighted[2]),
        }

    normal_packets = total_packets - abnormal_packets
    return risk_score, abnormal_packets, normal_packets, severity_split


def train_model(
    training_rows: list[dict[str, object]], group_config: dict[str, object]
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """Train RandomForest model and return metrics (with k-fold cross-validation).

    Pipeline:
    1. Hold-out split: 78% train, 22% test (stratified)
    2. K-fold cross-validation (k=5) on the training set for robust evaluation
    3. Train final model on full training set (for prediction use)
    4. Evaluate final model on the held-out test set

    Returns (model, metrics_dict) where metrics_dict includes:
      - accuracy: hold-out test accuracy
      - cv_mean_accuracy: mean accuracy across k folds
      - cv_std_accuracy: standard deviation across k folds
      - cv_fold_accuracies: per-fold accuracy list
      - report: classification_report on test set
      - hardware: HardwareMonitor summary
    """
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

    # ── K-Fold Cross-Validation ────────────────────────────────────
    cv_folds = 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=int(group_config["training_seed"]))
    cv_model = RandomForestClassifier(**rf_params)
    logger.info(
        f"Running {cv_folds}-fold cross-validation"
        f" (n_jobs={n_jobs})..."
    )
    cv_scores = cross_val_score(
        cv_model, x_train, y_train,
        cv=cv, scoring="accuracy", n_jobs=n_jobs,
    )
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    logger.info(
        f"CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}"
        f" (folds: {[round(s, 4) for s in cv_scores]})"
    )

    # ── Final model training & hold-out evaluation ─────────────────
    logger.info(f"Training final RandomForest model (n_jobs={n_jobs})...")

    monitor = HardwareMonitor(interval=0.5)
    monitor.start()

    model = RandomForestClassifier(**rf_params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    hw_summary = monitor.stop()

    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4, output_dict=True)

    logger.info(f"Model accuracy (hold-out): {accuracy:.4f}")

    return model, {
        "accuracy": round(accuracy, 4),
        "cv_mean_accuracy": round(cv_mean, 4),
        "cv_std_accuracy": round(cv_std, 4),
        "cv_folds": cv_folds,
        "cv_fold_accuracies": [round(float(s), 4) for s in cv_scores],
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
        "group_description": group_config.get("group_description", ""),
        "rf_params": dict(group_config["rf_params"]),
        "rf_params_desc": dict(group_config.get("rf_params_desc", {})),
        "data_profile": merged_data_profile(group_config),
        "data_profile_desc": dict(group_config.get("data_profile_desc", {})),
        "case_blueprints_desc": [
            {"case_name": bp.get("case_name", ""), "scenario_desc": bp.get("scenario_desc", "")}
            for bp in group_config.get("case_blueprints", [])
        ],
        "training_rows": len(training_rows),
        "training_label_counts": to_counts(training_rows, "label"),
        "window_count": len(input_cases),
        "bus_counts": to_counts(input_cases, "bus"),
        "expected_counts": to_counts(input_cases, "expected_label"),
        "predicted_counts": to_counts(predicted_rows, "predicted_label"),
        "accuracy": float(metrics["accuracy"]),
        "cv_folds": metrics.get("cv_folds", 0),
        "cv_mean_accuracy": metrics.get("cv_mean_accuracy", 0),
        "cv_std_accuracy": metrics.get("cv_std_accuracy", 0),
        "cv_fold_accuracies": metrics.get("cv_fold_accuracies", []),
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
    """Format summary as readable text with Chinese labels."""

    def _cn_label(key: str, cn_map: dict[str, str]) -> str:
        return f"{cn_map.get(key, key)}（{key}）" if key in cn_map else key

    rf_desc = dict(summary.get("rf_params_desc", {}))
    rf_lines = []
    for key, value in dict(summary["rf_params"]).items():
        label = _cn_label(key, RF_PARAMS_CN)
        desc = rf_desc.get(key, "")
        suffix = f"  ── {desc}" if desc else ""
        rf_lines.append(f"  - {label}: {value}{suffix}")

    dp_desc = dict(summary.get("data_profile_desc", {}))
    data_lines = []
    for key, value in dict(summary["data_profile"]).items():
        label = _cn_label(key, DATA_PROFILE_CN)
        desc = dp_desc.get(key, "")
        suffix = f"  ── {desc}" if desc else ""
        data_lines.append(f"  - {label}: {value}{suffix}")

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

    # 预测场景描述（如果有）
    case_desc_lines = []
    for bp in summary.get("case_blueprints_desc", []):
        desc = bp.get("scenario_desc", "")
        if desc:
            case_desc_lines.append(f"  [{bp['case_name']}] {desc}")

    parts = [
        f"分组：{summary['group_name']}",
        f"主题：{summary['theme']}",
    ]
    group_desc = summary.get("group_description", "")
    if group_desc:
        parts.append(f"场景说明：{group_desc}")
    parts.extend([
        "",
        "随机森林参数：",
        *rf_lines,
        "",
        "数据多样性参数：",
        *data_lines,
        "",
        f"训练样本总数：{summary['training_rows']:,}",
        "训练类别分布：",
        *training_lines,
        "",
        f"输入窗口数量：{summary['window_count']}",
        "总线分布：",
        *bus_lines,
        "预期标签分布：",
        *expected_lines,
        "预测标签分布：",
        *predicted_lines,
    ])
    if case_desc_lines:
        parts.extend(["", "预测场景时序说明：", *case_desc_lines])
    parts.extend([
        "",
        f"模型准确率 (hold-out)：{summary['accuracy']:.4f}",
        f"交叉验证准确率 ({summary.get('cv_folds', 0)}-fold)：{summary.get('cv_mean_accuracy', 0):.4f} ± {summary.get('cv_std_accuracy', 0):.4f}",
        f"置信度区间：{summary['confidence']['min']:.4f} / {summary['confidence']['mean']:.4f} / {summary['confidence']['max']:.4f}",
        f"风险分数区间：{summary['risk_score']['min']:.4f} / {summary['risk_score']['mean']:.4f} / {summary['risk_score']['max']:.4f}",
        f"误判窗口数：{summary['misclassified_count']}",
    ])

    return "\n".join(parts)
