"""CANary configuration constants for anomaly detection system."""

FEATURES = [
    "flow_kbps_mean",
    "frame_rate",
    "id_entropy",
    "replay_ratio",
    "fuzzy_ratio",
    "spoof_ratio",
    "error_ratio",
    "inter_arrival_cv",
    "burst_score",
    "uds_ratio",
    "payload_entropy",
]

LABELS = ["安全", "DoS", "重放", "模糊", "欺骗", "UDS非法会话"]

FEATURES_CN = {
    "flow_kbps_mean": "平均流量(kbps)",
    "frame_rate": "帧率",
    "id_entropy": "ID熵值",
    "replay_ratio": "重放比例",
    "fuzzy_ratio": "模糊比例",
    "spoof_ratio": "欺骗比例",
    "error_ratio": "错误比例",
    "inter_arrival_cv": "到达间隔变异",
    "burst_score": "突发分数",
    "uds_ratio": "UDS比例",
    "payload_entropy": "载荷熵值",
}

FEATURE_BOUNDS = {
    "flow_kbps_mean": (80.0, 560.0),
    "frame_rate": (500.0, 1900.0),
    "id_entropy": (0.08, 0.98),
    "replay_ratio": (0.0, 0.95),
    "fuzzy_ratio": (0.0, 0.95),
    "spoof_ratio": (0.0, 0.95),
    "error_ratio": (0.0, 0.35),
    "inter_arrival_cv": (0.05, 0.95),
    "burst_score": (0.05, 0.98),
    "uds_ratio": (0.0, 0.95),
    "payload_entropy": (0.08, 0.98),
}

CLASS_PROFILES = {
    "安全": {
        "flow_kbps_mean": (138.0, 18.0),
        "frame_rate": (710.0, 48.0),
        "id_entropy": (0.72, 0.07),
        "replay_ratio": (0.02, 0.01),
        "fuzzy_ratio": (0.01, 0.01),
        "spoof_ratio": (0.02, 0.01),
        "error_ratio": (0.02, 0.01),
        "inter_arrival_cv": (0.18, 0.06),
        "burst_score": (0.24, 0.08),
        "uds_ratio": (0.04, 0.02),
        "payload_entropy": (0.42, 0.08),
    },
    "DoS": {
        "flow_kbps_mean": (470.0, 36.0),
        "frame_rate": (1660.0, 95.0),
        "id_entropy": (0.24, 0.07),
        "replay_ratio": (0.05, 0.02),
        "fuzzy_ratio": (0.03, 0.02),
        "spoof_ratio": (0.06, 0.03),
        "error_ratio": (0.21, 0.05),
        "inter_arrival_cv": (0.82, 0.07),
        "burst_score": (0.9, 0.06),
        "uds_ratio": (0.04, 0.02),
        "payload_entropy": (0.2, 0.06),
    },
    "重放": {
        "flow_kbps_mean": (228.0, 24.0),
        "frame_rate": (920.0, 70.0),
        "id_entropy": (0.34, 0.08),
        "replay_ratio": (0.68, 0.08),
        "fuzzy_ratio": (0.05, 0.03),
        "spoof_ratio": (0.08, 0.03),
        "error_ratio": (0.08, 0.03),
        "inter_arrival_cv": (0.55, 0.08),
        "burst_score": (0.58, 0.08),
        "uds_ratio": (0.18, 0.05),
        "payload_entropy": (0.28, 0.08),
    },
    "模糊": {
        "flow_kbps_mean": (286.0, 28.0),
        "frame_rate": (1060.0, 84.0),
        "id_entropy": (0.9, 0.05),
        "replay_ratio": (0.07, 0.03),
        "fuzzy_ratio": (0.74, 0.08),
        "spoof_ratio": (0.06, 0.03),
        "error_ratio": (0.22, 0.05),
        "inter_arrival_cv": (0.66, 0.08),
        "burst_score": (0.64, 0.08),
        "uds_ratio": (0.12, 0.05),
        "payload_entropy": (0.92, 0.05),
    },
    "欺骗": {
        "flow_kbps_mean": (264.0, 24.0),
        "frame_rate": (980.0, 78.0),
        "id_entropy": (0.58, 0.08),
        "replay_ratio": (0.09, 0.04),
        "fuzzy_ratio": (0.05, 0.03),
        "spoof_ratio": (0.71, 0.07),
        "error_ratio": (0.14, 0.04),
        "inter_arrival_cv": (0.5, 0.08),
        "burst_score": (0.57, 0.08),
        "uds_ratio": (0.18, 0.05),
        "payload_entropy": (0.55, 0.08),
    },
    "UDS非法会话": {
        "flow_kbps_mean": (244.0, 22.0),
        "frame_rate": (900.0, 68.0),
        "id_entropy": (0.46, 0.08),
        "replay_ratio": (0.11, 0.04),
        "fuzzy_ratio": (0.04, 0.03),
        "spoof_ratio": (0.12, 0.04),
        "error_ratio": (0.1, 0.03),
        "inter_arrival_cv": (0.46, 0.07),
        "burst_score": (0.5, 0.08),
        "uds_ratio": (0.8, 0.07),
        "payload_entropy": (0.63, 0.07),
    },
}

FEATURE_ADJUSTMENTS = {
    "安全": {
        "flow_kbps_mean": -12.0,
        "frame_rate": -26.0,
        "error_ratio": -0.008,
        "burst_score": -0.03,
    },
    "DoS": {
        "flow_kbps_mean": 80.0,
        "frame_rate": 180.0,
        "id_entropy": -0.08,
        "error_ratio": 0.05,
        "inter_arrival_cv": 0.08,
        "burst_score": 0.1,
        "payload_entropy": -0.04,
    },
    "重放": {
        "replay_ratio": 0.1,
        "uds_ratio": 0.04,
        "inter_arrival_cv": 0.05,
        "payload_entropy": -0.04,
    },
    "模糊": {
        "fuzzy_ratio": 0.09,
        "id_entropy": 0.03,
        "error_ratio": 0.03,
        "payload_entropy": 0.03,
        "burst_score": 0.04,
    },
    "欺骗": {
        "spoof_ratio": 0.09,
        "burst_score": 0.05,
        "replay_ratio": 0.02,
        "error_ratio": 0.02,
    },
    "UDS非法会话": {
        "uds_ratio": 0.09,
        "replay_ratio": 0.03,
        "frame_rate": -28.0,
        "payload_entropy": 0.03,
    },
}

CONFUSION_MAP = {
    "安全": ["重放", "欺骗"],
    "DoS": ["欺骗", "模糊"],
    "重放": ["UDS非法会话", "安全"],
    "模糊": ["欺骗", "DoS"],
    "欺骗": ["模糊", "DoS"],
    "UDS非法会话": ["重放", "欺骗"],
}

DOMINANT_FEATURE = {
    "安全": None,
    "DoS": "burst_score",
    "重放": "replay_ratio",
    "模糊": "fuzzy_ratio",
    "欺骗": "spoof_ratio",
    "UDS非法会话": "uds_ratio",
}

DATA_PROFILE_DEFAULTS = {
    "noise_scale": 1.2,
    "boundary_ratio": 0.14,
    "drift_ratio": 0.1,
    "label_noise_ratio": 0.02,
}


def validate_domain_config() -> list[str]:
    """Validate feature, label, and profile consistency.

    Returns a list of human-readable issues. An empty list means the domain
    configuration is self-consistent enough for training and reporting.
    """
    issues: list[str] = []

    for feature in FEATURES:
        if feature not in FEATURE_BOUNDS:
            issues.append(f"Missing FEATURE_BOUNDS for feature: {feature}")
        if feature not in FEATURES_CN:
            issues.append(f"Missing FEATURES_CN mapping for feature: {feature}")

    for label in LABELS:
        if label not in CLASS_PROFILES:
            issues.append(f"Missing CLASS_PROFILES entry for label: {label}")
        if label not in FEATURE_ADJUSTMENTS:
            issues.append(f"Missing FEATURE_ADJUSTMENTS entry for label: {label}")
        if label not in CONFUSION_MAP:
            issues.append(f"Missing CONFUSION_MAP entry for label: {label}")

    for label, profile in CLASS_PROFILES.items():
        missing_features = [feature for feature in FEATURES if feature not in profile]
        if missing_features:
            issues.append(
                f"CLASS_PROFILES[{label!r}] missing features: {', '.join(missing_features)}"
            )

    for label, adjustments in FEATURE_ADJUSTMENTS.items():
        unknown_features = [feature for feature in adjustments if feature not in FEATURES]
        if unknown_features:
            issues.append(
                f"FEATURE_ADJUSTMENTS[{label!r}] references unknown features: {', '.join(unknown_features)}"
            )

    for label, neighbors in CONFUSION_MAP.items():
        unknown_neighbors = [neighbor for neighbor in neighbors if neighbor not in LABELS]
        if unknown_neighbors:
            issues.append(
                f"CONFUSION_MAP[{label!r}] references unknown labels: {', '.join(unknown_neighbors)}"
            )

    return issues
