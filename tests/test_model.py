"""Unit tests for model.py — ML pipeline (training, CV, metrics)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import (
    compute_case_metrics,
    format_run_summary,
    labels_from_rows,
    load_model,
    make_run_summary,
    matrix_from_rows,
    predict_cases,
    save_model,
    train_model,
)
from config import FEATURES, LABELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rows() -> list[dict[str, object]]:
    """Generate a small set of rows for matrix/labels testing."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(30):
        row: dict[str, object] = {}
        for f in FEATURES:
            from config import FEATURE_BOUNDS
            low, high = FEATURE_BOUNDS[f]
            row[f] = round(float(rng.uniform(low, high)), 4)
        row["label"] = LABELS[i % len(LABELS)]
        rows.append(row)
    return rows


@pytest.fixture
def mini_group_config() -> dict:
    return {
        "group_id": "test-grp",
        "group_name": "测试组",
        "theme": "测试",
        "training_seed": 42,
        "samples_per_label": 30,
        "rf_params": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
    }


@pytest.fixture
def training_rows() -> list[dict[str, object]]:
    from data_generator import generate_training_rows
    cfg = {
        "training_seed": 42,
        "samples_per_label": 20,
        "data_profile": {"label_noise_ratio": 0.0},
    }
    return generate_training_rows(cfg)


# ---------------------------------------------------------------------------
# matrix_from_rows
# ---------------------------------------------------------------------------

class TestMatrixFromRows:
    def test_shape(self, sample_rows):
        mat = matrix_from_rows(sample_rows)
        assert mat.shape == (30, 11)
        assert mat.dtype == np.float64

    def test_missing_feature_raises(self, sample_rows):
        del sample_rows[0]["flow_kbps_mean"]
        with pytest.raises(KeyError, match="Missing required feature columns"):
            matrix_from_rows(sample_rows)

    def test_data_integrity(self, sample_rows):
        mat = matrix_from_rows(sample_rows)
        # Check first row's first feature matches
        assert mat[0, 0] == pytest.approx(float(sample_rows[0][FEATURES[0]]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# labels_from_rows
# ---------------------------------------------------------------------------

class TestLabelsFromRows:
    def test_shape(self, sample_rows):
        labels = labels_from_rows(sample_rows, "label")
        assert len(labels) == 30

    def test_content(self, sample_rows):
        labels = labels_from_rows(sample_rows, "label")
        assert labels[0] == sample_rows[0]["label"]
        assert labels[-1] == sample_rows[-1]["label"]


# ---------------------------------------------------------------------------
# compute_case_metrics
# ---------------------------------------------------------------------------

class TestComputeCaseMetrics:
    @pytest.fixture
    def safe_row(self) -> dict[str, object]:
        return {
            "flow_kbps_mean": 138.0,
            "error_ratio": 0.02,
            "burst_score": 0.24,
            "inter_arrival_cv": 0.18,
            "replay_ratio": 0.02,
            "fuzzy_ratio": 0.01,
            "spoof_ratio": 0.02,
            "uds_ratio": 0.04,
            "total_packets": 1000,
        }

    @pytest.fixture
    def attack_row(self) -> dict[str, object]:
        return {
            "flow_kbps_mean": 470.0,
            "error_ratio": 0.21,
            "burst_score": 0.9,
            "inter_arrival_cv": 0.82,
            "replay_ratio": 0.05,
            "fuzzy_ratio": 0.03,
            "spoof_ratio": 0.06,
            "uds_ratio": 0.04,
            "total_packets": 1000,
        }

    def test_safe_prediction_severity_empty(self, safe_row):
        risk, abnormal, normal, severity = compute_case_metrics(
            safe_row, "安全", 0.9, 0.95
        )
        assert severity == {"low": 0, "medium": 0, "high": 0}
        assert 0 < risk < 1
        assert abnormal + normal == 1000

    def test_attack_prediction_has_severity(self, attack_row):
        risk, abnormal, normal, severity = compute_case_metrics(
            attack_row, "DoS", 0.95, 0.01
        )
        assert severity["high"] > 0 or severity["medium"] > 0
        assert risk > 0.18
        assert abnormal + normal == 1000

    def test_high_risk_severity_high_dominates(self, attack_row):
        # Force high risk via high confidence + attack features
        _, _, _, severity = compute_case_metrics(
            attack_row, "DoS", 0.99, 0.01
        )
        total_sev = severity["low"] + severity["medium"] + severity["high"]
        assert severity["high"] >= severity["low"]


# ---------------------------------------------------------------------------
# train_model (incl. cross-validation)
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_model_and_metrics(self, training_rows, mini_group_config):
        model, metrics = train_model(training_rows, mini_group_config)
        assert model is not None
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_cv_metrics_present(self, training_rows, mini_group_config):
        _, metrics = train_model(training_rows, mini_group_config)
        assert "cv_mean_accuracy" in metrics
        assert "cv_std_accuracy" in metrics
        assert "cv_fold_accuracies" in metrics
        assert metrics["cv_folds"] == 5
        assert len(metrics["cv_fold_accuracies"]) == 5
        assert 0 <= metrics["cv_mean_accuracy"] <= 1

    def test_cv_fold_accuracies_reasonable(self, training_rows, mini_group_config):
        _, metrics = train_model(training_rows, mini_group_config)
        for acc in metrics["cv_fold_accuracies"]:
            assert 0.5 <= acc <= 1.0, f"CV fold accuracy {acc} suspiciously low"


# ---------------------------------------------------------------------------
# predict_cases
# ---------------------------------------------------------------------------

class TestPredictCases:
    @staticmethod
    def _to_cases(rows: list[dict]) -> list[dict]:
        """Add required metadata fields to training rows so they can be used as cases."""
        cases = []
        for idx, row in enumerate(rows):
            case = dict(row)
            case["total_packets"] = int(round(float(row["flow_kbps_mean"]) * 20 * 0.55))
            case["timestamp"] = f"2026-03-23T09:{idx:02d}:00"
            case["expected_label"] = case.get("label", "安全")
            case["bus"] = "can0"
            case["window_id"] = f"W-{idx:03d}"
            case["case_name"] = f"test_case_{idx}"
            cases.append(case)
        return cases

    def test_output_length_matches_input(self, training_rows, mini_group_config):
        model, _ = train_model(training_rows, mini_group_config)
        cases = self._to_cases(training_rows[:5])
        predicted = predict_cases(model, cases)
        assert len(predicted) == 5

    def test_output_has_prediction_fields(self, training_rows, mini_group_config):
        model, _ = train_model(training_rows, mini_group_config)
        cases = self._to_cases(training_rows[:3])
        predicted = predict_cases(model, cases)
        for row in predicted:
            assert "predicted_label" in row
            assert "confidence" in row
            assert "risk_score" in row
            assert "abnormal_packets" in row
            assert "normal_packets" in row
            assert "severity_split" in row
            assert "probabilities" in row


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoadModel:
    def test_roundtrip(self, training_rows, mini_group_config):
        model, _ = train_model(training_rows, mini_group_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            save_model(model, path)
            loaded = load_model(path)
            assert loaded is not None
            # Verify it still predicts
            predicted = loaded.predict(matrix_from_rows(training_rows[:3]))
            assert len(predicted) == 3

    def test_load_nonexistent_returns_none(self):
        result = load_model(Path("/nonexistent/model.joblib"))
        assert result is None


# ---------------------------------------------------------------------------
# make_run_summary / format_run_summary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def test_summary_includes_cv_fields(self, training_rows, mini_group_config):
        model, metrics = train_model(training_rows, mini_group_config)
        cases = TestPredictCases._to_cases(training_rows[:5])
        predicted = predict_cases(model, cases)
        summary = make_run_summary(
            mini_group_config, training_rows, cases, predicted, metrics
        )
        assert "cv_folds" in summary
        assert summary["cv_folds"] == 5
        assert "cv_mean_accuracy" in summary
        assert "cv_std_accuracy" in summary

    def test_summary_misclassified_count(self, training_rows, mini_group_config):
        model, metrics = train_model(training_rows, mini_group_config)
        cases = TestPredictCases._to_cases(training_rows[:5])
        predicted = predict_cases(model, cases)
        summary = make_run_summary(
            mini_group_config, training_rows, cases, predicted, metrics
        )
        assert "misclassified_count" in summary
        assert isinstance(summary["misclassified_count"], int)

    def test_format_run_summary_returns_string(self, training_rows, mini_group_config):
        model, metrics = train_model(training_rows, mini_group_config)
        cases = TestPredictCases._to_cases(training_rows[:5])
        predicted = predict_cases(model, cases)
        summary = make_run_summary(
            mini_group_config, training_rows, cases, predicted, metrics
        )
        text = format_run_summary(summary)
        assert isinstance(text, str)
        assert "交叉验证" in text
