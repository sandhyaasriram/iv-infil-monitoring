"""
Workstream B: Temporal Infiltration-State Estimation
=====================================================
Compares five model families on synthetic simulator traces:
    1. Threshold        -- slope-based on temperature + pressure window
    2. Logistic Regression  -- hand-crafted window features (stat moments)
    3. Random Forest        -- hand-crafted window features
    4. DeepResNet1D         -- deep residual temporal CNN (existing baseline)
    5. TCN1D                -- Temporal Convolutional Network (causal, dilated)

KEY DESIGN DECISIONS
--------------------
Window labels use the per-timestep `label` column (0=none, 1=early,
2=moderate, 3=severe).  The window label = max label within the window.
This means:
  - Pre-onset windows in any trial         → label 0
  - Windows spanning/after early onset     → label 1
  - etc.
This captures infiltration onset timing rather than trial-level severity
category, which is the physically correct classification target.

BUG FIXED (vs first version):
  The previous run used label_int = severity.map(LABEL_MAP), which assigns
  a constant label to every window in a trial regardless of onset timing.
  Pre-onset windows in infiltrating trials then look identical to normal
  trial windows (same signal profile) but carry different labels, causing
  neural models to collapse to majority-class prediction (macro-F1 ≈ 0.103).

Evaluation protocol:
    - Trial-disjoint split with seed=42 (same 120/30/50 as prior experiments).
    - Window = 50s at 1 Hz. Stride = 25 on train (50% overlap), non-overlapping
      on val/test for unbiased window-level evaluation.
    - Per-channel z-score normalisation fit on training windows only.
    - Threshold tuned on validation trials only.
    - Neural: early stopping on val macro-F1, AdamW, cosine-LR schedule.

Detection delay (CORRECTED):
    - Raw delay = t_detection − t_onset  (can be negative = premature alarm).
    - Clamped delay = max(0, raw_delay).
    - Premature detection count reported explicitly.
    - NO clamping used when reporting mean_delay_raw_s.

Outputs: results/workstream_b/
"""

import os
import sys
import json
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & global config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.splits import split_iv_trials
from src.models.baselines import (
    build_logistic_regression_baseline,
    build_random_forest_baseline,
    DeepResNet1D,
)
from src.models.tcn import build_tcn

DATA_PATH   = BASE_DIR / "synthetic_iv_infiltration_dataset.csv"
RESULTS_DIR = BASE_DIR / "results" / "workstream_b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN    = 50        # default window (seconds at 1 Hz)
STRIDE_TRAIN  = 25        # 50 % overlap for training diversity
STRIDE_EVAL   = WINDOW_LEN  # non-overlapping on val / test

BATCH_SIZE    = 128
LR            = 5e-4
MAX_EPOCHS    = 80
PATIENCE      = 15
DEVICE        = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Use per-timestep `label` column (0–3) — NOT derived from severity.
    # The simulator writes the actual instantaneous infiltration state here.
    assert "label" in df.columns, "Expected 'label' column in dataset"
    # Confounder flag: trials where condition != 'normal' AND severity == 'none'
    # (sensor perturbation applied to a non-infiltrating trial)
    df["is_confounder"] = (
        (df["condition"] != "normal") & (df["severity"] == "none")
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Window extraction — uses per-timestep label column
# ─────────────────────────────────────────────────────────────────────────────
def extract_windows(trial_df: pd.DataFrame, window_len: int, stride: int,
                    channels: list):
    """
    Sliding-window extraction for ONE trial.

    Window label = max(label_ts[start:start+window_len]).
    This ensures any post-onset infiltration within the window is captured.

    Returns
    -------
    windows : (N_win, C, L)  float32
    labels  : (N_win,)       int64  — per-window label
    times   : (N_win,)       float64 — time_s of window START
    """
    arr      = trial_df.sort_values("time_s")
    signal   = arr[channels].values.T.astype(np.float32)   # (C, T)
    labels_ts = arr["label"].values.astype(np.int64)        # per-timestep label
    times_ts  = arr["time_s"].values
    T         = signal.shape[1]

    windows, labels, times = [], [], []
    for start in range(0, T - window_len + 1, stride):
        windows.append(signal[:, start : start + window_len])
        labels.append(int(labels_ts[start : start + window_len].max()))
        times.append(float(times_ts[start]))

    if not windows:
        return (np.empty((0, len(channels), window_len), np.float32),
                np.empty(0, np.int64),
                np.empty(0, np.float64))
    return (np.array(windows, np.float32),
            np.array(labels, np.int64),
            np.array(times, np.float64))


def build_window_dataset(trials_df: pd.DataFrame, window_len: int, stride: int,
                          channels: list):
    """Concatenate window arrays across all trials in trials_df."""
    all_w, all_l, all_t, all_tid = [], [], [], []
    for tid, grp in trials_df.groupby("trial_id"):
        w, l, t = extract_windows(grp, window_len, stride, channels)
        if len(w) == 0:
            continue
        all_w.append(w)
        all_l.append(l)
        all_t.append(t)
        all_tid.extend([tid] * len(w))

    if not all_w:
        empty = np.empty(0)
        return empty, empty, empty, np.array([])
    return (np.concatenate(all_w, 0),
            np.concatenate(all_l, 0),
            np.concatenate(all_t, 0),
            np.array(all_tid))


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel normalisation (fit on training data only)
# ─────────────────────────────────────────────────────────────────────────────
def normalise_windows(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """
    Per-channel z-score normalisation, NaN-safe.

    Some conditions (missing_temp, missing_pressure) produce NaN sensor readings.
    We use nanmean/nanstd to compute fit statistics and then replace remaining
    NaNs with channel mean (zero after normalisation) before returning arrays.

    X shape: (N, C, L)
    """
    N_tr, C, L = X_train.shape
    # Compute per-channel statistics ignoring NaN
    ch_mean = np.nanmean(X_train, axis=(0, 2), keepdims=True)   # (1, C, 1)
    ch_std  = np.nanstd( X_train, axis=(0, 2), keepdims=True).clip(1e-8)

    def _norm_impute(X):
        out = ((X - ch_mean) / ch_std).astype(np.float32)
        # Replace NaN (from missing sensor) with 0 (= channel mean post-norm)
        out = np.nan_to_num(out, nan=0.0)
        return out

    return _norm_impute(X_train), _norm_impute(X_val), _norm_impute(X_test), (ch_mean, ch_std)


def apply_norm(X: np.ndarray, norm_params) -> np.ndarray:
    ch_mean, ch_std = norm_params
    out = ((X - ch_mean) / ch_std).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted features for tabular models
# ─────────────────────────────────────────────────────────────────────────────
def windows_to_features(windows: np.ndarray) -> np.ndarray:
    """
    (N, C, L) → (N, C*5) statistical features.
    Per channel: mean, std, min, max, linear slope.
    """
    N, C, L = windows.shape
    t_vec  = np.arange(L, dtype=np.float32)
    t_norm = t_vec - t_vec.mean()
    denom  = float(np.dot(t_norm, t_norm)) or 1.0

    rows = np.empty((N, C * 5), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            sig = windows[i, c]
            base = c * 5
            rows[i, base + 0] = sig.mean()
            rows[i, base + 1] = sig.std()
            rows[i, base + 2] = sig.min()
            rows[i, base + 3] = sig.max()
            rows[i, base + 4] = float(np.dot(t_norm, sig)) / denom
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Threshold baseline
# ─────────────────────────────────────────────────────────────────────────────
class ThresholdDetector:
    """
    Binary threshold on temperature slope (drops during infiltration) and/or
    pressure slope (rises during infiltration).

    Threshold values tuned on validation windows by sweeping percentile grid
    and maximising balanced accuracy for binary (infiltration vs not) task.
    Predicts class 1 if BOTH applicable triggers fire; class 0 otherwise.
    """

    def __init__(self, channels: list):
        self.channels    = channels
        self.temp_thresh  = 0.0
        self.press_thresh = 0.0
        self.has_temp  = "temperature_c" in channels
        self.has_press = "pressure_adc"  in channels

    def _slopes(self, windows: np.ndarray):
        N, C, L = windows.shape
        t_norm = np.arange(L, dtype=np.float32)
        t_norm -= t_norm.mean()
        denom  = float(np.dot(t_norm, t_norm)) or 1.0
        ts = np.zeros(N); ps = np.zeros(N)
        for ci, ch in enumerate(self.channels):
            s = np.array([np.dot(t_norm, windows[i, ci]) / denom for i in range(N)])
            if ch == "temperature_c": ts = s
            elif ch == "pressure_adc": ps = s
        return ts, ps

    def _predict(self, ts, ps, tt, pt):
        N = len(ts)
        out = np.zeros(N, int)
        for i in range(N):
            t_trig = (ts[i] < tt) if self.has_temp  else True
            p_trig = (ps[i] > pt) if self.has_press else True
            out[i] = 1 if (t_trig and p_trig) else 0
        return out

    def fit(self, X_val: np.ndarray, y_val: np.ndarray):
        ts, ps = self._slopes(X_val)
        y_bin  = (y_val >= 1).astype(int)
        t_cands = np.percentile(ts, np.arange(5, 96, 10)) if self.has_temp  else [0.0]
        p_cands = np.percentile(ps, np.arange(5, 96, 10)) if self.has_press else [0.0]
        best_ba, best_tt, best_pt = 0.0, np.median(ts) if self.has_temp else 0.0, np.median(ps) if self.has_press else 0.0
        for tt in t_cands:
            for pt in p_cands:
                preds = self._predict(ts, ps, tt, pt)
                ba = balanced_accuracy_score(y_bin, preds)
                if ba > best_ba:
                    best_ba, best_tt, best_pt = ba, tt, pt
        self.temp_thresh  = best_tt
        self.press_thresh = best_pt

    def predict(self, windows: np.ndarray) -> np.ndarray:
        ts, ps = self._slopes(windows)
        return self._predict(ts, ps, self.temp_thresh, self.press_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# Neural training
# ─────────────────────────────────────────────────────────────────────────────
def train_neural(model: nn.Module, X_train, y_train, X_val, y_val) -> nn.Module:
    model = model.to(DEVICE)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
    yva_np = y_val

    # Class-balanced loss
    counts  = np.bincount(y_train, minlength=4).astype(np.float32)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
    crit    = nn.CrossEntropyLoss(weight=weights)

    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_f1, best_state, no_imp = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vpreds = model(Xva).argmax(1).cpu().numpy()
        vf1 = f1_score(yva_np, vpreds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1   = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_neural(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Detection delay (no clamping in raw metric)
# ─────────────────────────────────────────────────────────────────────────────
def compute_detection_delays(test_df: pd.DataFrame,
                              preds: np.ndarray,
                              times: np.ndarray,
                              tids:  np.ndarray,
                              min_consecutive: int = 2) -> dict:
    """
    Detection delay per infiltration trial.

    delay_raw    = t_detection − t_onset  (negative → premature alarm)
    delay_clamped = max(0, delay_raw)

    Missed trials receive a penalty = time remaining from onset to trial end.
    """
    delays_raw, delays_clamped = [], []
    n_detected = n_missed = n_premature = 0
    per_trial  = []

    for tid, grp in test_df.groupby("trial_id"):
        grp = grp.sort_values("time_s")
        sev = grp["severity"].iloc[0]
        if sev == "none":
            continue

        onset_rows = grp[grp["label"] >= 1]
        if len(onset_rows) == 0:
            continue
        t_onset = float(onset_rows["time_s"].iloc[0])

        mask     = tids == tid
        t_w      = times[mask]
        p_w      = preds[mask]
        order    = np.argsort(t_w)
        t_w, p_w = t_w[order], p_w[order]

        det_time = None
        for i in range(len(p_w) - min_consecutive + 1):
            if np.all(p_w[i : i + min_consecutive] >= 1):
                det_time = t_w[i]
                break

        if det_time is None:
            n_missed += 1
            pen = float(grp["time_s"].iloc[-1] - t_onset)
            delays_raw.append(pen)
            delays_clamped.append(pen)
            per_trial.append({
                "trial_id": int(tid), "severity": sev,
                "t_onset": t_onset, "detected": False,
                "delay_raw_s": pen, "delay_clamped_s": pen, "premature": False,
            })
        else:
            dr = float(det_time - t_onset)
            dc = max(0.0, dr)
            n_detected += 1
            if dr < 0:
                n_premature += 1
            delays_raw.append(dr)
            delays_clamped.append(dc)
            per_trial.append({
                "trial_id": int(tid), "severity": sev,
                "t_onset": t_onset, "t_detected": float(det_time),
                "detected": True,
                "delay_raw_s": round(dr, 2), "delay_clamped_s": round(dc, 2),
                "premature": dr < 0,
            })

    return {
        "n_infiltration_trials": n_detected + n_missed,
        "n_detected":             n_detected,
        "n_missed":               n_missed,
        "n_premature_alarms":     n_premature,
        "mean_delay_raw_s":       round(float(np.mean(delays_raw)),    2) if delays_raw    else None,
        "median_delay_raw_s":     round(float(np.median(delays_raw)),  2) if delays_raw    else None,
        "mean_delay_clamped_s":   round(float(np.mean(delays_clamped)),2) if delays_clamped else None,
        "per_trial":              per_trial,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confounder FPR
# ─────────────────────────────────────────────────────────────────────────────
def compute_confounder_fpr(test_df: pd.DataFrame,
                            preds: np.ndarray,
                            tids:  np.ndarray,
                            win_labels: np.ndarray) -> dict:
    """False-positive rate on windows from confounder trials (label=0, is_confounder=1)."""
    conf_tids = set(
        test_df.groupby("trial_id")["is_confounder"].first()
        .pipe(lambda s: s[s == 1].index.tolist())
    )
    mask     = np.array([t in conf_tids for t in tids]) & (win_labels == 0)
    n_total  = int(mask.sum())
    if n_total == 0:
        return {"fpr": None, "n_confounder_windows": 0, "n_false_alarms": 0,
                "note": "no confounder windows in test set"}
    n_fp = int(((preds >= 1) & mask).sum())
    fpr  = n_fp / n_total
    ci_hw = 1.96 * np.sqrt(fpr * (1 - fpr) / n_total)
    return {
        "fpr":                  round(float(fpr), 4),
        "n_confounder_windows": n_total,
        "n_false_alarms":       n_fp,
        "ci_lower":             round(max(0.0, fpr - ci_hw), 4),
        "ci_upper":             round(min(1.0, fpr + ci_hw), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full model evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, preds, y_true, times, tids, test_df):
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2, 3]).tolist()
    return {
        "model":            name,
        "macro_f1":         round(float(f1_score(y_true, preds, average="macro",     zero_division=0)), 4),
        "weighted_f1":      round(float(f1_score(y_true, preds, average="weighted",  zero_division=0)), 4),
        "balanced_acc":     round(float(balanced_accuracy_score(y_true, preds)), 4),
        "confusion_matrix": cm,
        "detection_delay":  compute_detection_delays(test_df, preds, times, tids),
        "confounder_fpr":   compute_confounder_fpr(test_df, preds, tids, y_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment: all 5 models, configurable channels & window length
# ─────────────────────────────────────────────────────────────────────────────
def run_main_experiment(df, channels, window_len=WINDOW_LEN,
                         stride_train=STRIDE_TRAIN, stride_eval=STRIDE_EVAL,
                         seeds=(42, 101, 202)):
    splits   = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
    train_df = splits["train_df"]
    val_df   = splits["val_df"]
    test_df  = splits["test_df"]

    print(f"    Split: {train_df['trial_id'].nunique()} train / "
          f"{val_df['trial_id'].nunique()} val / "
          f"{test_df['trial_id'].nunique()} test trials")

    X_tr, y_tr, t_tr, tid_tr = build_window_dataset(train_df, window_len, stride_train,  channels)
    X_va, y_va, t_va, tid_va = build_window_dataset(val_df,   window_len, stride_eval,   channels)
    X_te, y_te, t_te, tid_te = build_window_dataset(test_df,  window_len, stride_eval,   channels)

    print(f"    Windows: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")
    print(f"    Train label dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    X_tr_n, X_va_n, X_te_n, norm_p = normalise_windows(X_tr, X_va, X_te)
    n_ch = len(channels)
    results = {}

    # ── Threshold ─────────────────────────────────────────────────────────────
    print("    [Threshold]", end=" ", flush=True)
    thr = ThresholdDetector(channels)
    thr.fit(X_va_n, y_va)
    thr_preds = thr.predict(X_te_n)
    results["threshold"] = evaluate_model("threshold", thr_preds, y_te, t_te, tid_te, test_df)
    m = results["threshold"]
    print(f"macro_f1={m['macro_f1']:.3f}, bal_acc={m['balanced_acc']:.3f}, "
          f"delay_raw={m['detection_delay']['mean_delay_raw_s']}s")

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("    [LR]", end=" ", flush=True)
    Ftr = windows_to_features(X_tr_n)
    Fte = windows_to_features(X_te_n)
    lr  = build_logistic_regression_baseline()
    lr.fit(Ftr, y_tr)
    lr_preds = lr.predict(Fte)
    results["logistic_regression"] = evaluate_model("logistic_regression", lr_preds, y_te, t_te, tid_te, test_df)
    m = results["logistic_regression"]
    print(f"macro_f1={m['macro_f1']:.3f}, bal_acc={m['balanced_acc']:.3f}, "
          f"delay_raw={m['detection_delay']['mean_delay_raw_s']}s")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("    [RF]", end=" ", flush=True)
    rf = build_random_forest_baseline()
    rf.fit(Ftr, y_tr)
    rf_preds = rf.predict(Fte)
    results["random_forest"] = evaluate_model("random_forest", rf_preds, y_te, t_te, tid_te, test_df)
    m = results["random_forest"]
    print(f"macro_f1={m['macro_f1']:.3f}, bal_acc={m['balanced_acc']:.3f}, "
          f"delay_raw={m['detection_delay']['mean_delay_raw_s']}s")

    # ── DeepResNet & TCN (multi-seed) ─────────────────────────────────────────
    for mname, mfac in [
        ("deep_resnet", lambda: DeepResNet1D(in_channels=n_ch, num_classes=4)),
        ("tcn",         lambda: build_tcn(in_channels=n_ch, num_classes=4)),
    ]:
        seed_metrics = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"    [{mname} seed={seed}]", end=" ", flush=True)
            model = train_neural(mfac(), X_tr_n, y_tr, X_va_n, y_va)
            preds = predict_neural(model, X_te_n)
            sm    = evaluate_model(mname, preds, y_te, t_te, tid_te, test_df)
            seed_metrics.append(sm)
            print(f"macro_f1={sm['macro_f1']:.3f}, bal_acc={sm['balanced_acc']:.3f}, "
                  f"delay_raw={sm['detection_delay']['mean_delay_raw_s']}s")

        if len(seed_metrics) == 1:
            results[mname] = seed_metrics[0]
        else:
            f1s  = [s["macro_f1"]   for s in seed_metrics]
            bas  = [s["balanced_acc"] for s in seed_metrics]
            dels = [s["detection_delay"]["mean_delay_raw_s"] for s in seed_metrics]
            fprs = [s["confounder_fpr"].get("fpr") or 0 for s in seed_metrics]
            agg  = deepcopy(seed_metrics[0])
            agg.update({
                "macro_f1_mean":     round(float(np.mean(f1s)),  4),
                "macro_f1_std":      round(float(np.std(f1s)),   4),
                "balanced_acc_mean": round(float(np.mean(bas)),  4),
                "delay_raw_mean_s":  round(float(np.mean(dels)), 2),
                "delay_raw_std_s":   round(float(np.std(dels)),  2),
                "fpr_mean":          round(float(np.mean(fprs)), 4),
                "fpr_std":           round(float(np.std(fprs)),  4),
                "per_seed": [
                    {"seed": s, "macro_f1": f1s[i], "delay_raw_s": dels[i]}
                    for i, s in enumerate(seeds)
                ],
            })
            results[mname] = agg

    return results, (X_te_n, y_te, t_te, tid_te, test_df, norm_p)


# ─────────────────────────────────────────────────────────────────────────────
# Sensor-subset ablation — TCN only
# ─────────────────────────────────────────────────────────────────────────────
def sensor_ablation(df, seeds=(42, 101, 202)):
    configs = [
        ("temp_only",     ["temperature_c"]),
        ("pressure_only", ["pressure_adc"]),
        ("temp_pressure", ["temperature_c", "pressure_adc"]),
    ]
    out = {}
    for name, chs in configs:
        print(f"    {name}: ", end="", flush=True)
        splits   = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
        X_tr, y_tr, _, _ = build_window_dataset(splits["train_df"], WINDOW_LEN, STRIDE_TRAIN, chs)
        X_va, y_va, _, _ = build_window_dataset(splits["val_df"],   WINDOW_LEN, STRIDE_EVAL,  chs)
        X_te, y_te, _, _ = build_window_dataset(splits["test_df"],  WINDOW_LEN, STRIDE_EVAL,  chs)
        X_tr_n, X_va_n, X_te_n, _ = normalise_windows(X_tr, X_va, X_te)
        f1s = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = train_neural(build_tcn(in_channels=len(chs)), X_tr_n, y_tr, X_va_n, y_va)
            preds = predict_neural(model, X_te_n)
            f1s.append(float(f1_score(y_te, preds, average="macro", zero_division=0)))
        out[name] = {"macro_f1_mean": round(np.mean(f1s), 4),
                     "macro_f1_std":  round(np.std(f1s),  4)}
        print(f"macro_f1={out[name]['macro_f1_mean']:.3f} ± {out[name]['macro_f1_std']:.3f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Window-length ablation — TCN only
# ─────────────────────────────────────────────────────────────────────────────
def window_length_ablation(df, window_lens=(10, 25, 50), seed=42):
    out = {}
    for wlen in window_lens:
        print(f"    window={wlen}s: ", end="", flush=True)
        splits   = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
        X_tr, y_tr, _, _ = build_window_dataset(splits["train_df"], wlen, max(1, wlen // 2), CHANNELS_FULL)
        X_va, y_va, _, _ = build_window_dataset(splits["val_df"],   wlen, wlen, CHANNELS_FULL)
        X_te, y_te, _, _ = build_window_dataset(splits["test_df"],  wlen, wlen, CHANNELS_FULL)
        X_tr_n, X_va_n, X_te_n, _ = normalise_windows(X_tr, X_va, X_te)
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_neural(build_tcn(in_channels=2), X_tr_n, y_tr, X_va_n, y_va)
        preds = predict_neural(model, X_te_n)
        f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
        out[str(wlen)] = {"macro_f1": round(f1, 4)}
        print(f"macro_f1={f1:.3f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Temporal-shuffle control — does order matter for TCN vs ResNet?
# ─────────────────────────────────────────────────────────────────────────────
def shuffle_ablation(df, seed=42):
    """
    Train with time-axis within each window randomly permuted.
    If TCN/ResNet relies on temporal order this should lower F1.
    BUG FIX: use X[i] = X[i][:, idx] (not X[i, :, idx]) to preserve C dim.
    """
    splits   = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
    X_tr, y_tr, _, _ = build_window_dataset(splits["train_df"], WINDOW_LEN, STRIDE_TRAIN, CHANNELS_FULL)
    X_va, y_va, _, _ = build_window_dataset(splits["val_df"],   WINDOW_LEN, STRIDE_EVAL,  CHANNELS_FULL)
    X_te, y_te, _, _ = build_window_dataset(splits["test_df"],  WINDOW_LEN, STRIDE_EVAL,  CHANNELS_FULL)
    X_tr_n, X_va_n, X_te_n, _ = normalise_windows(X_tr, X_va, X_te)

    rng = np.random.default_rng(seed)
    X_tr_shuf = X_tr_n.copy()
    for i in range(len(X_tr_shuf)):
        idx = rng.permutation(WINDOW_LEN)
        X_tr_shuf[i] = X_tr_shuf[i][:, idx]   # (C, L) → shuffle L axis

    out = {}
    for mname, mfac in [("tcn", lambda: build_tcn(in_channels=2)),
                          ("deep_resnet", lambda: DeepResNet1D(in_channels=2))]:
        print(f"    {mname} shuffled: ", end="", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_neural(mfac(), X_tr_shuf, y_tr, X_va_n, y_va)
        preds = predict_neural(model, X_te_n)
        f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
        out[mname] = {"macro_f1": round(f1, 4)}
        print(f"macro_f1={f1:.3f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 20 % scarcity ablation
# ─────────────────────────────────────────────────────────────────────────────
def scarcity_ablation(df, seeds=(42, 101, 202)):
    splits     = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
    train_tids = splits["train_trials"]
    val_df     = splits["val_df"]
    test_df    = splits["test_df"]

    X_va, y_va, _, _           = build_window_dataset(val_df,  WINDOW_LEN, STRIDE_EVAL,  CHANNELS_FULL)
    X_te, y_te, t_te, tid_te   = build_window_dataset(test_df, WINDOW_LEN, STRIDE_EVAL,  CHANNELS_FULL)

    out = {}
    for seed in seeds:
        rng      = np.random.default_rng(seed)
        n_scarce = max(4, int(len(train_tids) * 0.20))
        scarce   = rng.choice(train_tids, size=n_scarce, replace=False)
        sdf      = df[df["trial_id"].isin(scarce)]
        X_tr, y_tr, _, _ = build_window_dataset(sdf, WINDOW_LEN, STRIDE_TRAIN, CHANNELS_FULL)
        X_tr_n, X_va_n, X_te_n, _ = normalise_windows(X_tr, X_va, X_te)

        for mname, is_neural in [("rf", False), ("tcn", True)]:
            print(f"    seed={seed}, {mname} ({n_scarce} trials): ", end="", flush=True)
            if is_neural:
                torch.manual_seed(seed); np.random.seed(seed)
                model = train_neural(build_tcn(in_channels=2), X_tr_n, y_tr, X_va_n, y_va)
                preds = predict_neural(model, X_te_n)
            else:
                rf = build_random_forest_baseline()
                rf.fit(windows_to_features(X_tr_n), y_tr)
                preds = rf.predict(windows_to_features(X_te_n))
            f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
            ba = float(balanced_accuracy_score(y_te, preds))
            out[f"{mname}_seed{seed}"] = {
                "macro_f1": round(f1, 4), "balanced_acc": round(ba, 4),
                "n_train_trials": n_scarce,
            }
            print(f"macro_f1={f1:.3f}")

    # Aggregate RF and TCN across seeds
    for mname in ["rf", "tcn"]:
        f1s = [out[f"{mname}_seed{s}"]["macro_f1"] for s in seeds]
        out[f"{mname}_summary"] = {
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std":  round(float(np.std(f1s)),  4),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario breakdown
# ─────────────────────────────────────────────────────────────────────────────
def scenario_breakdown(df, tcn_model, norm_params):
    """Evaluate trained TCN on each condition subset of the test set."""
    splits  = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
    test_df = splits["test_df"]
    out = {}
    for cond in ["normal", "noisy_temp", "noisy_pressure", "missing_temp", "missing_pressure"]:
        sub = test_df[test_df["condition"] == cond]
        if len(sub) == 0:
            continue
        X_c, y_c, _, _ = build_window_dataset(sub, WINDOW_LEN, STRIDE_EVAL, CHANNELS_FULL)
        if len(y_c) == 0:
            continue
        X_c_n = apply_norm(X_c, norm_params)
        preds = predict_neural(tcn_model, X_c_n)
        out[cond] = {
            "n_trials":    int(sub["trial_id"].nunique()),
            "n_windows":   int(len(y_c)),
            "macro_f1":    round(float(f1_score(y_c, preds, average="macro", zero_division=0)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y_c, preds)), 4),
        }
        print(f"    {cond:<22}: macro_f1={out[cond]['macro_f1']:.3f}, n_windows={len(y_c)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
CHANNELS_FULL = ["temperature_c", "pressure_adc"]


def main():
    print("=" * 70)
    print("WORKSTREAM B -- Temporal Infiltration-State Estimation")
    print("=" * 70)
    print("NOTE: Window labels use per-timestep `label` column (0=none,")
    print("      1=early, 2=moderate, 3=severe). Window label = max in window.")
    print("      This captures onset timing, not just trial-level severity.")

    df = load_dataset()
    print(f"\nDataset: {len(df)} rows, {df['trial_id'].nunique()} trials")

    all_results = {}

    # ── [1] Main experiment ────────────────────────────────────────────────────
    print("\n[1] Main experiment: temp+pressure, window=50s, seeds=[42,101,202]")
    main_res, (X_te_n, y_te, t_te, tid_te, test_df_main, norm_p) = run_main_experiment(
        df, CHANNELS_FULL, seeds=(42, 101, 202)
    )
    all_results["main"] = main_res

    # ── [2] Sensor ablation ────────────────────────────────────────────────────
    print("\n[2] Sensor ablation (TCN, 3 seeds)...")
    all_results["sensor_ablation"] = sensor_ablation(df, seeds=(42, 101, 202))

    # ── [3] Window-length ablation ─────────────────────────────────────────────
    print("\n[3] Window-length ablation (TCN, seed=42)...")
    all_results["window_ablation"] = window_length_ablation(df)

    # ── [4] Temporal-shuffle control ──────────────────────────────────────────
    print("\n[4] Temporal shuffle control (TCN + ResNet, seed=42)...")
    all_results["shuffle_ablation"] = shuffle_ablation(df, seed=42)

    # ── [5] 20 % scarcity ablation ────────────────────────────────────────────
    print("\n[5] 20%% scarcity ablation (TCN vs RF, 3 seeds)...")
    all_results["scarcity_20pct"] = scarcity_ablation(df, seeds=(42, 101, 202))

    # ── [6] Per-scenario breakdown ─────────────────────────────────────────────
    print("\n[6] Per-scenario breakdown (TCN seed=42)...")
    # Retrain TCN seed=42 for scenario breakdown
    splits   = split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42)
    X_tr_s, y_tr_s, _, _ = build_window_dataset(splits["train_df"], WINDOW_LEN, STRIDE_TRAIN, CHANNELS_FULL)
    X_va_s, y_va_s, _, _ = build_window_dataset(splits["val_df"],   WINDOW_LEN, STRIDE_EVAL,  CHANNELS_FULL)
    X_tr_sn, X_va_sn, _, norm_s = normalise_windows(X_tr_s, X_va_s, X_tr_s[:1])
    torch.manual_seed(42); np.random.seed(42)
    tcn_s42 = train_neural(build_tcn(in_channels=2), X_tr_sn, y_tr_s, X_va_sn, y_va_s)
    all_results["scenario_breakdown"] = scenario_breakdown(df, tcn_s42, norm_s)

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[7] Saving results...")
    save = deepcopy(all_results)
    for mkey in save.get("main", {}):
        dd = save["main"][mkey].get("detection_delay", {})
        dd.pop("per_trial", None)   # keep file manageable

    with open(RESULTS_DIR / "workstream_b_metrics.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"    Saved: {RESULTS_DIR / 'workstream_b_metrics.json'}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MAIN RESULTS (temp+pressure, window=50s, seeds=[42,101,202])")
    print("=" * 70)
    print(f"{'Model':<22} {'MacroF1':>9} {'BalAcc':>8} {'Delay_raw(s)':>13} {'FPR':>7}")
    print("-" * 65)
    for mname in ["threshold", "logistic_regression", "random_forest",
                  "deep_resnet", "tcn"]:
        m = main_res.get(mname, {})
        if not m: continue
        f1_v   = m.get("macro_f1_mean",  m.get("macro_f1",   0))
        f1_std = m.get("macro_f1_std", 0)
        ba_v   = m.get("balanced_acc_mean", m.get("balanced_acc", 0))
        dr_v   = m.get("detection_delay", {}).get("mean_delay_raw_s", "n/a")
        fpr_v  = m.get("confounder_fpr", {}).get("fpr")
        std_s  = f"+-{f1_std:.3f}" if f1_std > 0 else "       "
        fpr_s  = f"{fpr_v:.3f}" if fpr_v is not None else "  n/a"
        print(f"  {mname:<20} {f1_v:>7.3f}{std_s:>9}  {ba_v:>7.3f}  {str(dr_v):>12}  {fpr_s:>6}")

    print("\nSensor ablation (TCN):")
    for k, v in all_results["sensor_ablation"].items():
        print(f"  {k:<22}: {v['macro_f1_mean']:.3f} +/- {v['macro_f1_std']:.3f}")

    print("\nWindow-length ablation (TCN):")
    for k, v in all_results["window_ablation"].items():
        print(f"  window={k}s: {v['macro_f1']:.3f}")

    print("\nShuffle control (temporal order test):")
    for mname, v in all_results["shuffle_ablation"].items():
        orig = main_res.get(mname, {}).get("macro_f1_mean", main_res.get(mname, {}).get("macro_f1", 0))
        print(f"  {mname:<14}: original={orig:.3f}, shuffled={v['macro_f1']:.3f}, "
              f"drop={orig - v['macro_f1']:.3f}")

    print("\n20% scarcity ablation (3 seeds):")
    sc = all_results["scarcity_20pct"]
    for mname in ["rf", "tcn"]:
        s = sc.get(f"{mname}_summary", {})
        print(f"  {mname}: {s.get('macro_f1_mean', 0):.3f} +/- {s.get('macro_f1_std', 0):.3f}")

    print("\nPer-scenario breakdown (TCN seed=42):")
    for cond, v in all_results["scenario_breakdown"].items():
        print(f"  {cond:<24}: macro_f1={v['macro_f1']:.3f}, bal_acc={v['balanced_acc']:.3f}")

    print("\n" + "=" * 70)
    print("NOTE: All results from synthetic simulator data (N=200 trials).")
    print("      Detection delay reported RAW (negative = premature alarm).")
    print("      Confounder: non-normal condition trials with severity=none.")
    print("      A drop in F1 under shuffling indicates temporal structure use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
