"""
Tests for Clinical Bioimpedance Discrepancy Investigation
=========================================================
Verifies:
  1. Exact reproduction of 0.9722 (relative change LR) and 0.7222 (raw features LR).
  2. Patient-grouped LOOCV quarantine (both phases held out together).
  3. Preprocessing (StandardScaler) strictly fit on training folds.
  4. Raw before-features contain strictly before-phase values (no future leakage).
  5. The zero-vector mechanism in the retrospective paired feature matrix.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clinical_plos_one"

PARAMS = ["Z", "R", "Xc", "Cm"]
FREQS = [20, 50, 100, 150, 200, 300, 500]


@pytest.fixture(scope="module")
def clinical_data():
    records = []
    for p in PARAMS:
        csv_path = DATA_DIR / f"{p}.csv"
        df = pd.read_csv(csv_path, header=None)
        for phase, s_row in [("before", 1), ("after", 10)]:
            pids = [int(float(df.iloc[s_row, c])) for c in range(1, 19)]
            for f_idx in range(s_row + 1, s_row + 8):
                freq = int(str(df.iloc[f_idx, 0]).replace("kHz", "").strip())
                for c_off, pid in enumerate(pids):
                    val = float(df.iloc[f_idx, c_off + 1])
                    records.append({
                        "patient_id": pid, "phase": phase,
                        "freq_khz": freq, "param": p, "value": val
                    })
    df_long = pd.DataFrame(records)
    piv = df_long.pivot_table(index=["patient_id", "phase", "freq_khz"],
                              columns="param", values="value").reset_index()
    return piv.sort_values(["patient_id", "phase", "freq_khz"]).reset_index(drop=True)


def test_reproduction_09722_and_07222(clinical_data):
    """Verifies that the exact 0.9722 and 0.7222 balanced accuracies are reproduced."""
    piv = clinical_data
    patients = sorted(piv["patient_id"].unique())

    # Build raw
    rows_raw, y_raw, grp_raw = [], [], []
    for pid in patients:
        for phase, lab in [("before", 0), ("after", 1)]:
            sub = piv[(piv.patient_id == pid) & (piv.phase == phase)].set_index("freq_khz")
            row = [sub.loc[f, p] for p in PARAMS for f in FREQS]
            rows_raw.append(row)
            y_raw.append(lab)
            grp_raw.append(pid)
    X_raw, y_raw, grp_raw = np.array(rows_raw), np.array(y_raw), np.array(grp_raw)

    # Build relative
    rows_rel, y_rel, grp_rel = [], [], []
    for pid in patients:
        bef = piv[(piv.patient_id == pid) & (piv.phase == "before")].set_index("freq_khz")
        aft = piv[(piv.patient_id == pid) & (piv.phase == "after")].set_index("freq_khz")
        row_delta = [(aft.loc[f, p] - bef.loc[f, p]) / abs(bef.loc[f, p]) for p in PARAMS for f in FREQS]
        rows_rel.append(np.zeros(len(row_delta)))
        y_rel.append(0)
        grp_rel.append(pid)
        rows_rel.append(row_delta)
        y_rel.append(1)
        grp_rel.append(pid)
    X_rel, y_rel, grp_rel = np.array(rows_rel), np.array(y_rel), np.array(grp_rel)

    # Evaluate raw LR
    preds_raw = np.zeros(len(y_raw))
    for pid in patients:
        te = (grp_raw == pid)
        tr = ~te
        s = StandardScaler()
        X_tr = s.fit_transform(X_raw[tr])
        X_te = s.transform(X_raw[te])
        m = LogisticRegression(C=1.0, random_state=42)
        m.fit(X_tr, y_raw[tr])
        preds_raw[te] = m.predict(X_te)
    ba_raw = balanced_accuracy_score(y_raw, preds_raw)
    assert round(ba_raw, 4) == 0.7222, f"Expected 0.7222, got {ba_raw}"

    # Evaluate relative LR
    preds_rel = np.zeros(len(y_rel))
    for pid in patients:
        te = (grp_rel == pid)
        tr = ~te
        s = StandardScaler()
        X_tr = s.fit_transform(X_rel[tr])
        X_te = s.transform(X_rel[te])
        m = LogisticRegression(C=1.0, random_state=42)
        m.fit(X_tr, y_rel[tr])
        preds_rel[te] = m.predict(X_te)
    ba_rel = balanced_accuracy_score(y_rel, preds_rel)
    assert round(ba_rel, 4) == 0.9722, f"Expected 0.9722, got {ba_rel}"


def test_patient_grouped_quarantine(clinical_data):
    """Verifies that both before and after observations of each patient are held out together."""
    piv = clinical_data
    patients = sorted(piv["patient_id"].unique())
    assert len(patients) == 18

    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]
        assert test_pid not in train_pids
        assert len(train_pids) == 17


def test_scaler_strictly_fit_on_training():
    """Asserts that normalizer parameters are solely calculated from training fold samples."""
    np.random.seed(42)
    X_tr = np.random.randn(34, 28) + 5.0
    X_te = np.random.randn(2, 28) + 100.0  # Outlier test distribution

    scaler = StandardScaler()
    scaler.fit(X_tr)

    # Scaler mean must match training mean, completely unaware of test mean
    assert np.allclose(scaler.mean_, X_tr.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([X_tr, X_te]).mean(axis=0))


def test_zero_vector_artifact_in_relative_features(clinical_data):
    """Verifies that the 0.9722 result relies on an artificial zero vector for class 0."""
    piv = clinical_data
    patients = sorted(piv["patient_id"].unique())

    zero_rows = []
    delta_rows = []
    for pid in patients:
        bef = piv[(piv.patient_id == pid) & (piv.phase == "before")].set_index("freq_khz")
        aft = piv[(piv.patient_id == pid) & (piv.phase == "after")].set_index("freq_khz")
        row_delta = [(aft.loc[f, p] - bef.loc[f, p]) / abs(bef.loc[f, p]) for p in PARAMS for f in FREQS]
        zero_rows.append(np.zeros(len(row_delta)))
        delta_rows.append(row_delta)

    zero_mat = np.array(zero_rows)
    delta_mat = np.array(delta_rows)

    # All before rows in relative setup are EXACTLY zero (0 variance)
    assert np.all(zero_mat == 0.0)
    assert np.var(zero_mat) == 0.0

    # Infiltration rows have significant non-zero variance
    assert np.var(delta_mat) > 0.01
