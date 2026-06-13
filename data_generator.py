"""Synthetic training data generation."""

from __future__ import annotations

import numpy as np

from config import (
    CLASS_PROFILES,
    CONFUSION_MAP,
    DATA_PROFILE_DEFAULTS,
    DOMINANT_FEATURE,
    FEATURE_ADJUSTMENTS,
    FEATURE_BOUNDS,
    FEATURES,
    LABELS,
)


def merged_data_profile(group_config: dict[str, object]) -> dict[str, float]:
    """Merge user config with defaults."""
    profile = dict(DATA_PROFILE_DEFAULTS)
    profile.update(group_config.get("data_profile", {}))
    return {key: float(value) for key, value in profile.items()}


def dominant_weight(label: str, feature: str) -> float:
    """Get weight based on dominant feature."""
    dominant = DOMINANT_FEATURE[label]
    if dominant is None:
        return 0.0
    return 1.0 if dominant == feature else 0.32


def rng_like(feature: str, stress: float) -> float:
    """Generate stress-based adjustment."""
    phase = sum(ord(char) for char in feature) % 7
    return np.sin(stress * 4.2 + phase) + np.cos(stress * 2.8 + phase) * 0.4


def clamp_feature_local(feature: str, value: float) -> float:
    """Clamp feature value to valid range."""
    low, high = FEATURE_BOUNDS[feature]
    return float(np.clip(value, low, high))


def sample_feature_value(
    rng: np.random.Generator,
    label: str,
    feature: str,
    noise_scale: float,
    neighbor: str | None = None,
    blend_ratio: float = 0.0,
) -> float:
    """Sample a single feature value."""
    mean, std = CLASS_PROFILES[label][feature]
    if neighbor:
        neighbor_mean, neighbor_std = CLASS_PROFILES[neighbor][feature]
        mean = mean * (1.0 - blend_ratio) + neighbor_mean * blend_ratio
        std = ((std + neighbor_std) / 2.0) * (1.08 + blend_ratio * 0.6)
    return clamp_feature_local(feature, rng.normal(mean, std * noise_scale))


def apply_label_signature(
    row: dict[str, float], label: str, pressure: float, stress: float
) -> dict[str, float]:
    """Apply label-specific adjustments."""
    for feature, delta in FEATURE_ADJUSTMENTS.get(label, {}).items():
        row[feature] = clamp_feature_local(
            feature, row[feature] + delta * pressure * (0.78 + stress * 0.12)
        )

    row["flow_kbps_mean"] = clamp_feature_local(
        "flow_kbps_mean",
        row["flow_kbps_mean"] + (pressure - 1.0) * (22 if label == "安全" else 40),
    )
    row["frame_rate"] = clamp_feature_local(
        "frame_rate",
        row["frame_rate"] + (pressure - 1.0) * (110 if label == "安全" else 220),
    )
    row["error_ratio"] = clamp_feature_local(
        "error_ratio", row["error_ratio"] + max(0.0, pressure - 1.0) * 0.025
    )
    row["inter_arrival_cv"] = clamp_feature_local(
        "inter_arrival_cv",
        row["inter_arrival_cv"] + abs(pressure - 1.0) * 0.06,
    )

    attack_features = ["replay_ratio", "fuzzy_ratio", "spoof_ratio", "uds_ratio"]
    dominant = DOMINANT_FEATURE[label]
    for feature in attack_features:
        if label == "安全":
            row[feature] = clamp_feature_local(
                feature, row[feature] + rng_like(feature, stress) * 0.004
            )
            continue
        contamination = abs(stress) * 0.018 * dominant_weight(label, feature)
        row[feature] = clamp_feature_local(feature, row[feature] + contamination)
        if dominant and feature != dominant:
            row[feature] = clamp_feature_local(
                feature, row[feature] + abs(stress) * 0.01
            )

    return row


def normalize_training_row(
    row: dict[str, float], assigned_label: str
) -> dict[str, object]:
    """Normalize and format a training row."""
    normalized: dict[str, object] = {}
    for feature in FEATURES:
        value = clamp_feature_local(feature, row[feature])
        normalized[feature] = (
            int(round(value)) if feature == "frame_rate" else round(value, 4)
        )
    normalized["label"] = assigned_label
    return normalized


def generate_training_rows(group_config: dict[str, object]) -> list[dict[str, object]]:
    """Generate synthetic training data."""
    from utils import logger

    logger.info("Generating synthetic training data...")
    profile = merged_data_profile(group_config)
    rng = np.random.default_rng(int(group_config["training_seed"]))
    rows: list[dict[str, object]] = []
    samples_per_label = int(group_config["samples_per_label"])

    for label in LABELS:
        for _ in range(samples_per_label):
            scenario = rng.random()
            neighbor: str | None = None
            blend_ratio = 0.0
            noise_scale = profile["noise_scale"]

            if scenario < profile["boundary_ratio"]:
                neighbor = str(rng.choice(CONFUSION_MAP[label]))
                blend_ratio = float(rng.uniform(0.28, 0.56))
                noise_scale *= 1.22
            elif scenario < profile["boundary_ratio"] + profile["drift_ratio"]:
                neighbor = str(rng.choice(CONFUSION_MAP[label]))
                blend_ratio = float(rng.uniform(0.08, 0.24))
                noise_scale *= 1.1

            pressure = float(rng.normal(1.0, 0.1))
            stress = float(rng.normal(0.0, 0.8))
            feature_row = {
                feature: sample_feature_value(
                    rng, label, feature, noise_scale, neighbor=neighbor, blend_ratio=blend_ratio
                )
                for feature in FEATURES
            }
            feature_row = apply_label_signature(feature_row, label, pressure, stress)

            assigned_label = label
            if rng.random() < profile["label_noise_ratio"]:
                assigned_label = str(rng.choice(CONFUSION_MAP[label]))

            rows.append(normalize_training_row(feature_row, assigned_label))

    logger.info(f"Generated {len(rows)} training samples")
    return rows


def generate_dynamic_cases(
    group_config: dict[str, object],
) -> list[dict[str, object]]:
    """Dynamically generate prediction cases from case_blueprints.

    Instead of relying on static input_cases.json, this function synthesizes
    prediction windows at runtime using CLASS_PROFILES + blueprint parameters
    (intensity, packet_scale, jitter).  Each call produces slightly different
    feature values, enabling dynamic / non-deterministic prediction.
    """
    from datetime import datetime, timedelta

    from utils import logger

    blueprints = group_config.get("case_blueprints", [])
    if not blueprints:
        logger.warning("No case_blueprints found, returning empty list")
        return []

    seed = int(group_config.get("training_seed", 0)) + 9999
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 3, 23, 9, 0, 0)
    cases: list[dict[str, object]] = []

    for idx, bp in enumerate(blueprints):
        target_label = str(bp["target_label"])
        intensity = float(bp.get("intensity", 1.0))
        packet_scale = float(bp.get("packet_scale", 1.0))
        jitter = float(bp.get("jitter", 0.8))

        # Build feature values from CLASS_PROFILES with intensity modulation
        case: dict[str, object] = {
            "window_id": str(bp["window_id"]),
            "timestamp": str(bp.get(
                "timestamp",
                (base_time + timedelta(minutes=5 * idx)).isoformat(),
            )),
            "case_name": str(bp.get("case_name", target_label)),
            "bus": str(bp.get("bus", "can0")),
            "window_seconds": int(bp.get("window_seconds", 20)),
        }

        for feature in FEATURES:
            mean, std = CLASS_PROFILES[target_label][feature]
            # Apply intensity: push features toward/away from attack profile
            modulated_mean = mean * intensity + mean * (1.0 - intensity) * 0.5
            noise = rng.normal(0, std * jitter * 0.6)
            value = float(np.clip(modulated_mean + noise, *FEATURE_BOUNDS[feature]))
            # Round: integer for frame_rate, 4 decimals for others
            case[feature] = (
                int(round(value)) if feature == "frame_rate" else round(value, 4)
            )

        # Derive total_packets from flow rate
        flow = float(case["flow_kbps_mean"])
        window_sec = int(case["window_seconds"])
        case["total_packets"] = int(round(flow * window_sec * packet_scale * 0.55))
        case["expected_label"] = target_label

        cases.append(case)

    logger.info(f"Dynamically generated {len(cases)} prediction cases")
    return cases
