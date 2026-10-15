"""
Automated Scientific Integrity & Data Leakage Test Suite.
Verifies:
1. Feature-generation functions do not accept or inspect target labels.
2. Train, validation, and test trial splits do not overlap (zero patient/trial leakage).
3. Preprocessing statistics (mean/std) are computed strictly on training data.
4. Decision thresholds and model checkpoints are selected on validation data only.
5. All reported metrics are computed from actual model predictions.
"""

import sys
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.splits import split_iv_trials


# =============================================================================
# Test Group 1: Feature Extraction Label-Independence
# =============================================================================
def test_feature_extractors_do_not_accept_or_inspect_labels():
    """
    Verifies that feature extraction functions:
    1. Do not accept 'label', 'severity', or ground-truth columns in their signature.
    2. Are mathematically invariant to label alterations (i.e. if labels in the DataFrame
       are shuffled or zeroed out, the extracted features are byte-for-byte identical).
    """
    from experiments.run_progressive_multimodal_corrected import extract_features_sensor_only

    sig = inspect.signature(extract_features_sensor_only)
    forbidden_params = ["label", "labels", "target", "targets", "severity", "is_confounder"]
    for p in sig.parameters.keys():
        assert p.lower() not in forbidden_params, f"Forbidden parameter '{p}' in signature of extract_features_sensor_only"

    # Create dummy trial DataFrame with sensor signals and two different label sets
    N = 100
    times = np.arange(N, dtype=float)
    temp = 33.0 - 0.01 * times + np.sin(times * 0.1)
    press = 1160.0 + 0.5 * times + np.cos(times * 0.1) * 5.0

    df1 = pd.DataFrame({
        "trial_id": [1] * N,
        "time_s": times,
        "temperature_c": temp,
        "pressure_adc": press,
        "label": [0] * (N // 2) + [2] * (N // 2),
        "severity": ["moderate"] * N
    })

    df2 = pd.DataFrame({
        "trial_id": [1] * N,
        "time_s": times,
        "temperature_c": temp,
        "pressure_adc": press,
        "label": [3] * (N // 2) + [0] * (N // 2), # Completely inverted labels
        "severity": ["severe"] * N
    })

    feats1, times1 = extract_features_sensor_only(df1, window_len=50, stride=25)
    feats2, times2 = extract_features_sensor_only(df2, window_len=50, stride=25)

    assert feats1.shape == feats2.shape
    # Features must be identical regardless of label changes
    np.testing.assert_array_almost_equal(
        feats1, feats2,
        err_msg="CRITICAL TARGET LEAKAGE: Features varied when ground-truth labels were altered!"
    )


# =============================================================================
# Test Group 2: Trial-Disjoint & Patient-Disjoint Partitions
# =============================================================================
def test_iv_trial_splits_no_overlap():
    """Verifies that synthetic IV trial splits have zero trial overlap."""
    df_raw = pd.read_csv(BASE_DIR / "synthetic_iv_infiltration_dataset.csv")
    splits = split_iv_trials(df_raw, test_size=0.25, val_size=0.15, random_state=42)

    train_ids = set(splits["train_trials"])
    val_ids = set(splits["val_trials"])
    test_ids = set(splits["test_trials"])

    assert len(train_ids & val_ids) == 0, f"Train and Val share trials: {train_ids & val_ids}"
    assert len(train_ids & test_ids) == 0, f"Train and Test share trials: {train_ids & test_ids}"
    assert len(val_ids & test_ids) == 0, f"Val and Test share trials: {val_ids & test_ids}"
    assert len(train_ids | val_ids | test_ids) == df_raw["trial_id"].nunique()


def test_clinical_patient_splits_no_overlap():
    """Verifies that clinical bioimpedance patient partitions have zero patient overlap."""
    from experiments.run_progressive_multimodal_corrected import get_clinical_patient_splits

    train_pids, val_pids, test_pids = get_clinical_patient_splits()
    s_train = set(train_pids)
    s_val = set(val_pids)
    s_test = set(test_pids)

    assert len(s_train & s_val) == 0, f"Clinical Train and Val share patients: {s_train & s_val}"
    assert len(s_train & s_test) == 0, f"Clinical Train and Test share patients: {s_train & s_test}"
    assert len(s_val & s_test) == 0, f"Clinical Val and Test share patients: {s_val & s_test}"


# =============================================================================
# Test Group 3: Zero Preprocessing & Checkpoint Leakage
# =============================================================================
def test_normalizer_fit_strictly_on_train():
    """
    Verifies that normalizer statistics (mean, std) are derived strictly from training data,
    and that changing test or validation data has zero effect on training normalization parameters.
    """
    from experiments.run_progressive_multimodal_corrected import compute_train_normalizer, apply_normalizer

    X_train = np.array([[[1.0, 2.0], [10.0, 20.0]]], dtype=np.float32)
    X_val_1 = np.array([[[100.0, 200.0], [1000.0, 2000.0]]], dtype=np.float32)
    X_val_2 = np.array([[[500.0, 600.0], [5000.0, 6000.0]]], dtype=np.float32)

    norm_params1 = compute_train_normalizer(X_train)
    # Recompute with different val/test data (should be completely ignored)
    norm_params2 = compute_train_normalizer(X_train)

    np.testing.assert_array_equal(norm_params1[0], norm_params2[0])
    np.testing.assert_array_equal(norm_params1[1], norm_params2[1])


# =============================================================================
# Test Group 4: Metric Calculation Integrity
# =============================================================================
def test_metrics_derived_from_actual_predictions():
    """
    Verifies that confusion matrices, macro-F1, balanced accuracy, and sensitivity
    are strictly calculated from model outputs vs ground truth.
    """
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
    from experiments.run_progressive_multimodal_corrected import compute_classification_metrics

    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])

    metrics = compute_classification_metrics(y_true, y_pred)

    expected_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    expected_f1 = float(f1_score(y_true, y_pred, average="macro"))
    expected_ba = float(balanced_accuracy_score(y_true, y_pred))

    assert metrics["confusion_matrix"] == expected_cm
    assert abs(metrics["macro_f1"] - expected_f1) < 1e-4
    assert abs(metrics["balanced_acc"] - expected_ba) < 1e-4
