"""Unit tests for data_generator.py — synthetic data generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_generator import (
    apply_label_signature,
    clamp_feature_local,
    dominant_weight,
    generate_dynamic_cases,
    generate_training_rows,
    merged_data_profile,
    normalize_training_row,
    rng_like,
    sample_feature_value,
)
from config import (
    CLASS_PROFILES,
    DOMINANT_FEATURE,
    FEATURES,
    LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_group_config() -> dict:
    """Minimal group_config for data generation tests."""
    return {
        "group_id": "test-grp",
        "group_name": "测试组",
        "theme": "测试主题",
        "training_seed": 42,
        "samples_per_label": 20,
        "rf_params": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
        "data_profile": {
            "noise_scale": 1.0,
            "boundary_ratio": 0.1,
            "drift_ratio": 0.05,
            "label_noise_ratio": 0.0,
        },
        "case_blueprints": [
            {
                "window_id": "W-001",
                "case_name": "安全正常",
                "target_label": "安全",
                "bus": "can0",
                "window_seconds": 20,
            },
            {
                "window_id": "W-002",
                "case_name": "DoS攻击",
                "target_label": "DoS",
                "bus": "can1",
                "window_seconds": 20,
            },
        ],
    }


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# merged_data_profile
# ---------------------------------------------------------------------------

class TestMergedDataProfile:
    def test_merges_user_overrides(self, sample_group_config):
        profile = merged_data_profile(sample_group_config)
        # user data_profile has noise_scale=1.0, overrides default 1.2
        assert profile["noise_scale"] == 1.0

    def test_fallback_to_defaults(self):
        profile = merged_data_profile({})
        assert profile["label_noise_ratio"] == 0.02


# ---------------------------------------------------------------------------
# dominant_weight
# ---------------------------------------------------------------------------

class TestDominantWeight:
    def test_safe_returns_zero(self):
        assert dominant_weight("安全", "burst_score") == 0.0

    def test_dominant_feature_returns_one(self):
        assert dominant_weight("DoS", "burst_score") == 1.0
        assert dominant_weight("重放", "replay_ratio") == 1.0

    def test_non_dominant_returns_sub_weight(self):
        assert dominant_weight("DoS", "replay_ratio") == 0.32


# ---------------------------------------------------------------------------
# rng_like
# ---------------------------------------------------------------------------

class TestRngLike:
    def test_returns_float(self):
        val = rng_like("flow_kbps_mean", 1.0)
        assert isinstance(val, float)

    def test_same_input_same_output(self):
        assert rng_like("flow_kbps_mean", 0.0) == rng_like("flow_kbps_mean", 0.0)

    def test_range_bounded(self):
        for s in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            val = rng_like("flow_kbps_mean", s)
            assert -2.5 <= val <= 2.5, f"rng_like({s}) = {val} out of expected range"


# ---------------------------------------------------------------------------
# clamp_feature_local
# ---------------------------------------------------------------------------

class TestClampFeatureLocal:
    def test_within_bounds_unchanged(self):
        assert clamp_feature_local("flow_kbps_mean", 200.0) == 200.0

    def test_below_low_clamped(self):
        assert clamp_feature_local("flow_kbps_mean", 10.0) == 80.0

    def test_above_high_clamped(self):
        assert clamp_feature_local("flow_kbps_mean", 9999.0) == 560.0


# ---------------------------------------------------------------------------
# sample_feature_value
# ---------------------------------------------------------------------------

class TestSampleFeatureValue:
    def test_returns_float_in_bounds(self, fixed_rng):
        val = sample_feature_value(fixed_rng, "安全", "flow_kbps_mean", 1.0)
        assert isinstance(val, float)
        assert 80.0 <= val <= 560.0

    def test_with_neighbor_differs(self, fixed_rng):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        val_no_neighbor = sample_feature_value(rng1, "安全", "flow_kbps_mean", 1.0)
        val_with_neighbor = sample_feature_value(
            rng2, "安全", "flow_kbps_mean", 1.0,
            neighbor="DoS", blend_ratio=0.3,
        )
        # With blend, distributions differ — values should differ
        assert val_no_neighbor != val_with_neighbor


# ---------------------------------------------------------------------------
# apply_label_signature
# ---------------------------------------------------------------------------

class TestApplyLabelSignature:
    def test_applies_dos_adjustments(self):
        row = {f: float(CLASS_PROFILES["DoS"][f][0]) for f in FEATURES}
        adjusted = apply_label_signature(row, "DoS", 1.0, 0.0)
        # DoS adjustments push flow_kbps_mean up
        assert adjusted["flow_kbps_mean"] >= row["flow_kbps_mean"]

    def test_output_all_features_present(self):
        row = {f: float(CLASS_PROFILES["安全"][f][0]) for f in FEATURES}
        adjusted = apply_label_signature(row, "安全", 1.0, 0.0)
        assert set(adjusted.keys()) == set(FEATURES)

    def test_output_within_bounds(self):
        row = {f: float(CLASS_PROFILES["安全"][f][0]) for f in FEATURES}
        for pressure in [0.8, 1.0, 1.5]:
            adjusted = apply_label_signature(dict(row), "安全", pressure, 0.0)
            for f in FEATURES:
                from config import FEATURE_BOUNDS
                low, high = FEATURE_BOUNDS[f]
                assert low <= adjusted[f] <= high, (
                    f"Feature {f}={adjusted[f]} out of [{low},{high}] at pressure={pressure}"
                )


# ---------------------------------------------------------------------------
# normalize_training_row
# ---------------------------------------------------------------------------

class TestNormalizeTrainingRow:
    def test_includes_label(self):
        row = {f: float(CLASS_PROFILES["安全"][f][0]) for f in FEATURES}
        result = normalize_training_row(row, "安全")
        assert result["label"] == "安全"

    def test_frame_rate_is_int(self):
        row = {f: float(CLASS_PROFILES["DoS"][f][0]) for f in FEATURES}
        result = normalize_training_row(row, "DoS")
        assert isinstance(result["frame_rate"], int)


# ---------------------------------------------------------------------------
# generate_training_rows
# ---------------------------------------------------------------------------

class TestGenerateTrainingRows:
    def test_returns_correct_count(self, sample_group_config):
        rows = generate_training_rows(sample_group_config)
        expected = 6 * 20  # 6 labels × 20 per label
        assert len(rows) == expected

    def test_each_row_has_all_features(self, sample_group_config):
        rows = generate_training_rows(sample_group_config)
        for row in rows:
            for f in FEATURES:
                assert f in row
            assert "label" in row
            assert row["label"] in LABELS

    def test_label_distribution(self, sample_group_config):
        rows = generate_training_rows(sample_group_config)
        from collections import Counter
        counts = Counter(r["label"] for r in rows)
        # With label_noise_ratio=0.0, each label should have ~20 rows
        for label in LABELS:
            assert counts[label] == 20, f"Expected 20 rows for {label}, got {counts[label]}"


# ---------------------------------------------------------------------------
# generate_dynamic_cases
# ---------------------------------------------------------------------------

class TestGenerateDynamicCases:
    def test_returns_cases_from_blueprints(self, sample_group_config):
        cases = generate_dynamic_cases(sample_group_config)
        assert len(cases) == 2

    def test_case_has_expected_fields(self, sample_group_config):
        cases = generate_dynamic_cases(sample_group_config)
        for case in cases:
            assert "window_id" in case
            assert "timestamp" in case
            assert "case_name" in case
            assert "bus" in case
            assert "expected_label" in case
            assert "total_packets" in case
            for f in FEATURES:
                assert f in case

    def test_total_packets_positive(self, sample_group_config):
        cases = generate_dynamic_cases(sample_group_config)
        for case in cases:
            assert case["total_packets"] > 0

    def test_empty_blueprints_returns_empty(self):
        cfg = {"case_blueprints": [], "training_seed": 42}
        cases = generate_dynamic_cases(cfg)
        assert cases == []
