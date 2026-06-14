"""Unit tests for config.py — domain configuration validation."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLASS_PROFILES,
    CONFUSION_MAP,
    DOMINANT_FEATURE,
    FEATURE_ADJUSTMENTS,
    FEATURE_BOUNDS,
    FEATURES,
    FEATURES_CN,
    LABELS,
    validate_domain_config,
)


class TestDomainConfig:
    """Tests for the domain configuration constants and validator."""

    def test_feature_count(self):
        """FEATURES list should contain exactly 11 elements."""
        assert len(FEATURES) == 11, f"Expected 11 features, got {len(FEATURES)}"

    def test_label_count(self):
        """LABELS should contain 6 classes (1 safe + 5 attack types)."""
        assert len(LABELS) == 6, f"Expected 6 labels, got {len(LABELS)}"
        assert "安全" in LABELS

    def test_every_feature_has_bounds(self):
        """Each feature must have FEATURE_BOUNDS defined."""
        for f in FEATURES:
            assert f in FEATURE_BOUNDS, f"Missing bounds for {f}"
            low, high = FEATURE_BOUNDS[f]
            assert low < high, f"Invalid bounds for {f}: low={low}, high={high}"

    def test_every_feature_has_chinese_label(self):
        """Each feature must have a FEATURES_CN mapping."""
        for f in FEATURES:
            assert f in FEATURES_CN, f"Missing Chinese label for {f}"

    def test_every_label_has_class_profile(self):
        """Each label must have a CLASS_PROFILES entry with all features."""
        for label in LABELS:
            assert label in CLASS_PROFILES, f"Missing CLASS_PROFILES for {label}"
            for f in FEATURES:
                assert f in CLASS_PROFILES[label], (
                    f"CLASS_PROFILES[{label}] missing feature {f}"
                )
                mean, std = CLASS_PROFILES[label][f]
                assert std > 0, f"CLASS_PROFILES[{label}][{f}] std must be > 0"

    def test_every_label_has_adjustments(self):
        """Each label must have FEATURE_ADJUSTMENTS."""
        for label in LABELS:
            assert label in FEATURE_ADJUSTMENTS, (
                f"Missing FEATURE_ADJUSTMENTS for {label}"
            )

    def test_every_label_has_confusion_map(self):
        """Each label must have CONFUSION_MAP defined."""
        for label in LABELS:
            assert label in CONFUSION_MAP, f"Missing CONFUSION_MAP for {label}"
            for neighbor in CONFUSION_MAP[label]:
                assert neighbor in LABELS, (
                    f"CONFUSION_MAP[{label}] references unknown label {neighbor}"
                )

    def test_every_label_has_dominant_feature(self):
        """Each label must have DOMINANT_FEATURE mapping."""
        for label in LABELS:
            assert label in DOMINANT_FEATURE, (
                f"Missing DOMINANT_FEATURE for {label}"
            )
            dominant = DOMINANT_FEATURE[label]
            if dominant is not None:
                assert dominant in FEATURES, (
                    f"DOMINANT_FEATURE[{label}]={dominant} not in FEATURES"
                )

    def test_validate_domain_config_empty(self):
        """validate_domain_config() should return empty list for valid config."""
        issues = validate_domain_config()
        assert issues == [], f"Unexpected config issues: {issues}"


class TestFeatureBounds:
    """Tests for feature bound consistency."""

    def test_all_bounds_are_valid_ranges(self):
        """All feature bounds must be (low, high) with low < high."""
        for feature, (low, high) in FEATURE_BOUNDS.items():
            assert low < high, (
                f"Bounds for {feature}: low={low} must be < high={high}"
            )
            assert low >= 0, f"Low bound for {feature} should be >= 0"

    def test_flow_kbps_bounds(self):
        """flow_kbps_mean bounds should match expected CAN bus range."""
        low, high = FEATURE_BOUNDS["flow_kbps_mean"]
        assert low == 80.0
        assert high == 560.0

    def test_ratio_features_bounded_0_to_1(self):
        """All ratio features should be bounded in [0, ~1] range."""
        ratio_features = [
            "replay_ratio", "fuzzy_ratio", "spoof_ratio",
            "error_ratio", "uds_ratio",
        ]
        for f in ratio_features:
            low, high = FEATURE_BOUNDS[f]
            assert low == 0.0, f"{f} low bound should be 0.0"
            assert 0 < high <= 1.0, f"{f} high bound should be in (0, 1]"


class TestClassProfiles:
    """Tests for class profile distribution parameters."""

    def test_safe_has_low_attack_ratios(self):
        """安全 (safe) profile should have low attack-type ratios."""
        safe = CLASS_PROFILES["安全"]
        for attack_f in ["replay_ratio", "fuzzy_ratio", "spoof_ratio", "uds_ratio"]:
            mean, _ = safe[attack_f]
            assert mean < 0.06, (
                f"Safe profile {attack_f} mean={mean} should be < 0.06"
            )

    def test_dos_has_high_flow_and_burst(self):
        """DoS profile should have high flow rate and burst score."""
        dos = CLASS_PROFILES["DoS"]
        assert dos["flow_kbps_mean"][0] > 400, "DoS should have high flow"
        assert dos["burst_score"][0] > 0.8, "DoS should have high burst"

    def test_replay_has_high_replay_ratio(self):
        """重放 (Replay) profile should have elevated replay_ratio."""
        replay_mean = CLASS_PROFILES["重放"]["replay_ratio"][0]
        assert replay_mean > 0.5, "Replay profile should have high replay_ratio"

    def test_fuzzy_has_high_entropy(self):
        """模糊 (Fuzzy) profile should have high id_entropy and payload_entropy."""
        fuzzy = CLASS_PROFILES["模糊"]
        assert fuzzy["id_entropy"][0] > 0.8
        assert fuzzy["payload_entropy"][0] > 0.8
