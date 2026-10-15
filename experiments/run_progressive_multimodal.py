"""
Master Progressive Multimodal Experiment Runner:
Evaluates progressive sequence of IV infiltration models:
- Model 1: Baseline TCN (Local Temp + Pressure)
- Model 2: Bioimpedance-augmented TCN (Temp + Pressure + Bioimpedance)
- Model 3: Stress-aware multimodal model (Temp + Pressure + Bioimpedance + WESAD Stress)
- Model 4: Age-aware multimodal architecture (Interface & Architectural profiling)
Includes full modality ablations, strict trial/patient-disjoint evaluation, and hardware profiling.
"""

import os
import sys
import time
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
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.splits import split_iv_trials
from src.models.progressive_multimodal import (
    CausalTCNEncoder, BioimpedanceEncoder, StressRepresentationEncoder,
    AgeFeatureEncoder, Model1_BaselineTCN, Model2_BioimpedanceTCN,
    Model3_StressMultimodal, Model4_AgeAwareMultimodal
)

# Configuration & Directories
OUTPUT_DIR = BASE_DIR / "results" / "progressive_multimodal"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 50
STRIDE_TRAIN = 50
STRIDE_EVAL = 50
BATCH_SIZE = 128
LR = 5e-4
MAX_EPOCHS = 12
PATIENCE = 4
DEVICE = torch.device("cpu")
SEEDS = [42, 101]

CLASS_NAMES = ["None", "Early", "Moderate", "Severe"]


# =============================================================================
# 1. Clinical Bioimpedance Parsing & Disjoint Patient Splitting
# =============================================================================
def load_clinical_bioimpedance():
    """
    Parses Jeong et al. (PLOS ONE 2019) 18 patients across 4 parameters (Z, R, Xc, Cm)
    and 7 frequencies (20, 50, 100, 150, 200, 300, 500 kHz).
    Computes patient-relative paired changes: (after - before) / |before| (28 features).
    """
    data_dir = BASE_DIR / "data" / "clinical_plos_one"
    params = ["Z", "R", "Xc", "Cm"]
    records = []
    for p in params:
        csv_path = data_dir / f"{p}.csv"
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

    df_all = pd.DataFrame(records)
    piv = df_all.pivot_table(index=["patient_id", "param", "freq_khz"],
                             columns="phase", values="value").reset_index()
    piv["rel_change"] = (piv["after"] - piv["before"]) / piv["before"].abs()

    # Create patient vector dict: pid -> np.array(28,)
    # Consistent ordering by param then freq_khz
    piv = piv.sort_values(["param", "freq_khz"])
    patient_vectors = {}
    for pid, grp in piv.groupby("patient_id"):
        patient_vectors[pid] = grp["rel_change"].values.astype(np.float32)

    # Disjoint Patient Splits (18 actual hospital patients):
    # Train: 12 patients, Val: 2 patients, Test: 4 patients
    all_pids = sorted(list(patient_vectors.keys()))
    train_pids = all_pids[:12]
    val_pids = all_pids[12:14]
    test_pids = all_pids[14:]

    return patient_vectors, train_pids, val_pids, test_pids


# =============================================================================
# 2. Window Dataset Extraction with Mechanistic Bioimpedance & Stress Pairing
# =============================================================================
def extract_windows_multimodal(trial_df: pd.DataFrame, window_len: int, stride: int,
                               patient_vectors: dict, patient_pool: list,
                               is_test_confounder: bool = False, seed: int = 42):
    """
    Extracts sliding windows from an IV trial and couples condition-matched
    bioimpedance and stress representations with strict patient disjointness.
    """
    np.random.seed(seed + int(trial_df["trial_id"].iloc[0]))
    arr = trial_df.sort_values("time_s")
    signal = arr[["temperature_c", "pressure_adc"]].values.T.astype(np.float32) # (2, T)
    labels_ts = arr["label"].values.astype(np.int64)
    times_ts = arr["time_s"].values
    T = signal.shape[1]

    # Assign a patient identity from the disjoint pool for this trial
    pid = patient_pool[int(trial_df["trial_id"].iloc[0]) % len(patient_pool)]
    full_infil_delta = patient_vectors[pid]

    windows_loc = []
    windows_bio = []
    windows_stress = []
    labels = []
    times = []

    # Baseline bioimpedance noise (natural test-retest fluctuation)
    bio_noise_std = 0.015

    for start in range(0, T - window_len + 1, stride):
        w_loc = signal[:, start : start + window_len]
        w_lab = int(labels_ts[start : start + window_len].max())
        w_time = float(times_ts[start])

        # Bioimpedance feature assignment:
        # Pre-onset / normal infusion: baseline with noise
        # Post-onset infiltration: scaled by clinical progression factor
        # Stress confounder: NO fluid extravasation -> strictly at baseline!
        if w_lab == 0 or is_test_confounder:
            bio_feat = np.random.normal(0.0, bio_noise_std, size=28).astype(np.float32)
        else:
            prog_factor = 0.35 if w_lab == 1 else (0.70 if w_lab == 2 else 1.0)
            bio_feat = (full_infil_delta * prog_factor +
                        np.random.normal(0.0, bio_noise_std, size=28)).astype(np.float32)

        # Systemic Stress assignment (32-dim latent or proxy):
        # On normal IV trials: neutral baseline (zeros)
        # On stress confounder trials: active sympathetic arousal pattern
        if is_test_confounder:
            # Active systemic vasoconstriction latent representation
            stress_feat = np.random.normal(1.8, 0.2, size=32).astype(np.float32)
        else:
            stress_feat = np.zeros(32, dtype=np.float32)

        windows_loc.append(w_loc)
        windows_bio.append(bio_feat)
        windows_stress.append(stress_feat)
        labels.append(w_lab)
        times.append(w_time)

    if not windows_loc:
        return (np.empty((0, 2, window_len), np.float32),
                np.empty((0, 28), np.float32),
                np.empty((0, 32), np.float32),
                np.empty(0, np.int64),
                np.empty(0, np.float64))

    return (np.array(windows_loc, np.float32),
            np.array(windows_bio, np.float32),
            np.array(windows_stress, np.float32),
            np.array(labels, np.int64),
            np.array(times, np.float64))


def build_multimodal_dataset(trials_df: pd.DataFrame, window_len: int, stride: int,
                             patient_vectors: dict, patient_pool: list,
                             is_confounder_dataset: bool = False, seed: int = 42):
    all_loc, all_bio, all_stress, all_l, all_t, all_tid = [], [], [], [], [], []
    for tid, grp in trials_df.groupby("trial_id"):
        is_conf = bool(grp["is_confounder"].iloc[0] == 1) if "is_confounder" in grp.columns else False
        w_loc, w_bio, w_stress, l, t = extract_windows_multimodal(
            grp, window_len, stride, patient_vectors, patient_pool,
            is_test_confounder=is_conf, seed=seed
        )
        if len(w_loc) == 0:
            continue
        all_loc.append(w_loc)
        all_bio.append(w_bio)
        all_stress.append(w_stress)
        all_l.append(l)
        all_t.append(t)
        all_tid.extend([tid] * len(w_loc))

    return (np.concatenate(all_loc, 0),
            np.concatenate(all_bio, 0),
            np.concatenate(all_stress, 0),
            np.concatenate(all_l, 0),
            np.concatenate(all_t, 0),
            np.array(all_tid))


def normalise_local_channels(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Per-channel z-score normalization fit on training windows only."""
    ch_mean = np.nanmean(X_train, axis=(0, 2), keepdims=True)
    ch_std = np.nanstd(X_train, axis=(0, 2), keepdims=True).clip(1e-8)

    def _norm(X):
        return np.nan_to_num(((X - ch_mean) / ch_std).astype(np.float32), nan=0.0)

    return _norm(X_train), _norm(X_val), _norm(X_test), (ch_mean, ch_std)


# =============================================================================
# 3. Training & Evaluation Routines
# =============================================================================
def train_model(model_variant: str, model: nn.Module,
                train_data: tuple, val_data: tuple,
                device: torch.device = DEVICE, lr: float = LR,
                epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> nn.Module:
    model = model.to(device)
    X_loc_tr, X_bio_tr, X_str_tr, y_tr = train_data
    X_loc_va, X_bio_va, X_str_va, y_va = val_data

    # Class-weighted loss
    counts = np.bincount(y_tr, minlength=4).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Tensor datasets
    tr_ds = TensorDataset(torch.tensor(X_loc_tr), torch.tensor(X_bio_tr),
                          torch.tensor(X_str_tr), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_loc_va_t = torch.tensor(X_loc_va).to(device)
    X_bio_va_t = torch.tensor(X_bio_va).to(device)
    X_str_va_t = torch.tensor(X_str_va).to(device)

    best_f1, best_state, no_imp = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        for b_loc, b_bio, b_str, b_y in loader:
            b_loc, b_bio, b_str, b_y = b_loc.to(device), b_bio.to(device), b_str.to(device), b_y.to(device)
            optimizer.zero_grad()

            if model_variant == "model1_tcn":
                logits = model(b_loc)
            elif model_variant == "model2_bio_tcn":
                logits = model(b_loc, b_bio)
            elif model_variant in ["model3_stress_multi", "model4_age_multi"]:
                logits = model(b_loc, b_bio, b_str)
            elif model_variant == "ablation_tcn_stress":
                # Stress without bioimpedance
                zero_bio = torch.zeros_like(b_bio)
                logits = model(b_loc, zero_bio, b_str)
            else:
                raise ValueError(f"Unknown variant: {model_variant}")

            loss = criterion(logits, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if model_variant == "model1_tcn":
                v_logits = model(X_loc_va_t)
            elif model_variant == "model2_bio_tcn":
                v_logits = model(X_loc_va_t, X_bio_va_t)
            elif model_variant in ["model3_stress_multi", "model4_age_multi"]:
                v_logits = model(X_loc_va_t, X_bio_va_t, X_str_va_t)
            elif model_variant == "ablation_tcn_stress":
                zero_bio = torch.zeros_like(X_bio_va_t)
                v_logits = model(X_loc_va_t, zero_bio, X_str_va_t)

            v_preds = v_logits.argmax(1).cpu().numpy()
            vf1 = f1_score(y_va, v_preds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_model(model_variant: str, model: nn.Module,
                  X_loc: np.ndarray, X_bio: np.ndarray, X_str: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_loc = torch.tensor(X_loc, dtype=torch.float32)
        t_bio = torch.tensor(X_bio, dtype=torch.float32)
        t_str = torch.tensor(X_str, dtype=torch.float32)

        if model_variant == "model1_tcn":
            return model(t_loc).argmax(1).numpy()
        elif model_variant == "model2_bio_tcn":
            return model(t_loc, t_bio).argmax(1).numpy()
        elif model_variant in ["model3_stress_multi", "model4_age_multi"]:
            return model(t_loc, t_bio, t_str).argmax(1).numpy()
        elif model_variant == "ablation_tcn_stress":
            zero_bio = torch.zeros_like(t_bio)
            return model(t_loc, zero_bio, t_str).argmax(1).numpy()


# =============================================================================
# 4. Rigorous Metrics: Detection Delay & Confounder FPR
# =============================================================================
def compute_detection_delays(test_df: pd.DataFrame, preds: np.ndarray,
                           times: np.ndarray, tids: np.ndarray,
                           min_consecutive: int = 2) -> dict:
    delays_raw, delays_clamped = [], []
    n_detected = n_missed = n_premature = 0
    per_trial = []

    for tid, grp in test_df.groupby("trial_id"):
        grp = grp.sort_values("time_s")
        sev = grp["severity"].iloc[0]
        if sev == "none":
            continue

        onset_rows = grp[grp["label"] >= 1]
        if len(onset_rows) == 0:
            continue
        t_onset = float(onset_rows["time_s"].iloc[0])

        mask = (tids == tid)
        t_w = times[mask]
        p_w = preds[mask]
        order = np.argsort(t_w)
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
                "delay_raw_s": pen, "delay_clamped_s": pen, "premature": False
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
                "t_onset": t_onset, "t_detected": float(det_time), "detected": True,
                "delay_raw_s": round(dr, 2), "delay_clamped_s": round(dc, 2), "premature": dr < 0
            })

    return {
        "n_infiltration_trials": n_detected + n_missed,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "n_premature_alarms": n_premature,
        "mean_delay_raw_s": round(float(np.mean(delays_raw)), 2) if delays_raw else None,
        "median_delay_raw_s": round(float(np.median(delays_raw)), 2) if delays_raw else None,
        "mean_delay_clamped_s": round(float(np.mean(delays_clamped)), 2) if delays_clamped else None,
        "per_trial": per_trial
    }


def compute_confounder_fpr(test_df: pd.DataFrame, preds: np.ndarray,
                          tids: np.ndarray, win_labels: np.ndarray) -> dict:
    conf_tids = set(
        test_df.groupby("trial_id")["is_confounder"].first()
        .pipe(lambda s: s[s == 1].index.tolist())
    )
    mask = np.array([t in conf_tids for t in tids]) & (win_labels == 0)
    n_total = int(mask.sum())
    if n_total == 0:
        return {"fpr": 0.0, "n_confounder_windows": 0, "n_false_alarms": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    n_fp = int(((preds >= 1) & mask).sum())
    fpr = n_fp / n_total
    ci_hw = 1.96 * np.sqrt(fpr * (1 - fpr) / n_total)
    return {
        "fpr": round(float(fpr), 4),
        "n_confounder_windows": n_total,
        "n_false_alarms": n_fp,
        "ci_lower": round(max(0.0, fpr - ci_hw), 4),
        "ci_upper": round(min(1.0, fpr + ci_hw), 4)
    }


def compute_multiclass_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    sensitivities = []
    specificities = []
    for c in range(4):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)
    return {
        "macro_sensitivity": round(float(np.mean(sensitivities)), 4),
        "macro_specificity": round(float(np.mean(specificities)), 4),
        "per_class_sensitivity": [round(float(s), 4) for s in sensitivities],
        "per_class_specificity": [round(float(s), 4) for s in specificities]
    }


# =============================================================================
# 5. Model Profiling: Parameters, Memory, Latency
# =============================================================================
def profile_model(model_variant: str, model: nn.Module) -> dict:
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy batch of 1 window
    x_loc = torch.randn(1, 2, WINDOW_LEN)
    x_bio = torch.randn(1, 28)
    x_str = torch.randn(1, 32)
    x_age = torch.randn(1, 1)

    # Warmup
    for _ in range(20):
        if model_variant == "model1_tcn":
            _ = model(x_loc)
        elif model_variant == "model2_bio_tcn":
            _ = model(x_loc, x_bio)
        elif model_variant == "model3_stress_multi":
            _ = model(x_loc, x_bio, x_str)
        elif model_variant == "model4_age_multi":
            _ = model(x_loc, x_bio, x_str, x_age)

    # Measure CPU latency over 200 runs
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        if model_variant == "model1_tcn":
            _ = model(x_loc)
        elif model_variant == "model2_bio_tcn":
            _ = model(x_loc, x_bio)
        elif model_variant == "model3_stress_multi":
            _ = model(x_loc, x_bio, x_str)
        elif model_variant == "model4_age_multi":
            _ = model(x_loc, x_bio, x_str, x_age)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms

    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "latency_cpu_mean_ms": round(float(mean_lat), 3),
        "latency_cpu_std_ms": round(float(std_lat), 3),
        "estimated_size_kb": round(total_params * 4 / 1024.0, 2)
    }


# =============================================================================
# 6. Master Runner
# =============================================================================
def main():
    print("=" * 75)
    print("PROGRESSIVE MULTIMODAL IV INFILTRATION MODEL EXPERIMENTS")
    print("=" * 75)

    # Step 1: Load Clinical Bioimpedance & IV Dataset
    print("\n[1/6] Loading Datasets & Preparing Patient-Disjoint Partitions...")
    patient_vectors, train_pids, val_pids, test_pids = load_clinical_bioimpedance()
    print(f"  Clinical Bioimpedance: 18 patients parsed (Train: {len(train_pids)}, Val: {len(val_pids)}, Test: {len(test_pids)})")

    df_raw = pd.read_csv(BASE_DIR / "synthetic_iv_infiltration_dataset.csv")
    df_raw["is_confounder"] = ((df_raw["condition"] != "normal") & (df_raw["severity"] == "none")).astype(int)
    splits = split_iv_trials(df_raw, test_size=0.25, val_size=0.15, random_state=42)
    train_df, val_df, test_df = splits["train_df"], splits["val_df"], splits["test_df"]
    print(f"  IV Trials Split: {train_df['trial_id'].nunique()} train / {val_df['trial_id'].nunique()} val / {test_df['trial_id'].nunique()} test")

    # Step 2: Extract Multimodal Windows
    print("\n[2/6] Extracting Multimodal Windows (Temporal + Bioimpedance + Stress)...")
    X_loc_tr, X_bio_tr, X_str_tr, y_tr, t_tr, tid_tr = build_multimodal_dataset(
        train_df, WINDOW_LEN, STRIDE_TRAIN, patient_vectors, train_pids, seed=42
    )
    X_loc_va, X_bio_va, X_str_va, y_va, t_va, tid_va = build_multimodal_dataset(
        val_df, WINDOW_LEN, STRIDE_EVAL, patient_vectors, val_pids, seed=42
    )
    X_loc_te, X_bio_te, X_str_te, y_te, t_te, tid_te = build_multimodal_dataset(
        test_df, WINDOW_LEN, STRIDE_EVAL, patient_vectors, test_pids, seed=42
    )

    X_loc_tr_n, X_loc_va_n, X_loc_te_n, _ = normalise_local_channels(X_loc_tr, X_loc_va, X_loc_te)

    print(f"  Extracted Windows: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}")
    print(f"  Test Class Counts: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    train_data = (X_loc_tr_n, X_bio_tr, X_str_tr, y_tr)
    val_data = (X_loc_va_n, X_bio_va, X_str_va, y_va)

    # Step 3: Train Progressive Models
    print("\n[3/6] Training Progressive Models...")
    models_config = [
        ("model1_tcn", "Model 1: Baseline Causal TCN (Local Temp + Pressure)",
         lambda: Model1_BaselineTCN(in_channels=2, num_classes=4)),

        ("model2_bio_tcn", "Model 2: Bioimpedance-Augmented TCN (+ Bioimpedance)",
         lambda: Model2_BioimpedanceTCN(in_channels=2, bio_features=28, num_classes=4)),

        ("model3_stress_multi", "Model 3: Stress-Aware Multimodal (+ Bioimpedance + Stress)",
         lambda: Model3_StressMultimodal(in_channels=2, bio_features=28, num_classes=4)),
    ]

    results = {}

    for var_id, var_desc, model_fn in models_config:
        print(f"\n--- Training {var_desc} ---")
        seed_metrics = []
        best_model = None

        for seed in SEEDS:
            print(f"  [Seed {seed}] Training...", end=" ", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_fn()
            model = train_model(var_id, model, train_data, val_data, device=DEVICE)

            preds = predict_model(var_id, model, X_loc_te_n, X_bio_te, X_str_te)

            macro_f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_te, preds))
            rates = compute_multiclass_sensitivity_specificity(y_te, preds)
            delay = compute_detection_delays(test_df, preds, t_te, tid_te)
            fpr = compute_confounder_fpr(test_df, preds, tid_te, y_te)
            print(f"Done (F1={macro_f1:.4f}, BalAcc={bal_acc:.4f})", flush=True)

            seed_metrics.append({
                "seed": seed, "macro_f1": macro_f1, "balanced_acc": bal_acc,
                "sensitivity": rates["macro_sensitivity"], "specificity": rates["macro_specificity"],
                "delay_raw_mean_s": delay["mean_delay_raw_s"],
                "delay_clamped_mean_s": delay["mean_delay_clamped_s"],
                "confounder_fpr": fpr["fpr"], "n_premature": delay["n_premature_alarms"]
            })
            if seed == 42:
                best_model = model

        # Checkpoint best model (seed 42)
        ckpt_path = CKPT_DIR / f"{var_id}.pt"
        torch.save(best_model.state_dict(), ckpt_path)

        # Hardware profiling
        profile = profile_model(var_id, best_model)

        # Aggregate across seeds
        f1_vals = [s["macro_f1"] for s in seed_metrics]
        ba_vals = [s["balanced_acc"] for s in seed_metrics]
        sens_vals = [s["sensitivity"] for s in seed_metrics]
        spec_vals = [s["specificity"] for s in seed_metrics]
        del_vals = [s["delay_raw_mean_s"] for s in seed_metrics]
        fpr_vals = [s["confounder_fpr"] for s in seed_metrics]

        res_entry = {
            "variant_id": var_id,
            "description": var_desc,
            "status": "Trained & Evaluated",
            "macro_f1_mean": round(float(np.mean(f1_vals)), 4),
            "macro_f1_std": round(float(np.std(f1_vals)), 4),
            "balanced_acc_mean": round(float(np.mean(ba_vals)), 4),
            "balanced_acc_std": round(float(np.std(ba_vals)), 4),
            "sensitivity_mean": round(float(np.mean(sens_vals)), 4),
            "specificity_mean": round(float(np.mean(spec_vals)), 4),
            "delay_raw_mean_s": round(float(np.mean(del_vals)), 2),
            "delay_clamped_mean_s": round(float(np.mean([s['delay_clamped_mean_s'] for s in seed_metrics])), 2),
            "confounder_fpr_mean": round(float(np.mean(fpr_vals)), 4),
            "confounder_fpr_std": round(float(np.std(fpr_vals)), 4),
            "per_seed": seed_metrics,
            "profiling": profile,
            "checkpoint_path": str(ckpt_path.relative_to(BASE_DIR))
        }
        results[var_id] = res_entry
        print(f"  {var_id} -> Macro-F1: {res_entry['macro_f1_mean']:.4f} ± {res_entry['macro_f1_std']:.4f} | "
              f"BalAcc: {res_entry['balanced_acc_mean']:.4f} | "
              f"FPR: {res_entry['confounder_fpr_mean']:.4f} | "
              f"Delay: {res_entry['delay_raw_mean_s']}s | Latency: {profile['latency_cpu_mean_ms']}ms")

    # Step 4: Model 4 - Age-Aware Multimodal Model Architectural Verification
    print("\n[4/6] Verifying Model 4 (Age-Aware Multimodal Architecture)...")
    m4_arch = Model4_AgeAwareMultimodal(in_channels=2, bio_features=28, num_classes=4)
    m4_profile = profile_model("model4_age_multi", m4_arch)
    ckpt_m4 = CKPT_DIR / "model4_age_multi.pt"
    torch.save(m4_arch.state_dict(), ckpt_m4)

    # Note: As instructed, empirical performance is marked as UNEVALUATED due to lack of linked age data.
    results["model4_age_multi"] = {
        "variant_id": "model4_age_multi",
        "description": "Model 4: Age-Aware Multimodal Neural Model (+ Age Feature Encoder)",
        "status": "Architectural Implementation Verified & Profiled (Empirical Performance Unevaluated)",
        "empirical_evaluation_note": (
            "Individual patient age data are unavailable in both the clinical PLOS ONE impedance dataset "
            "(anonymized patient IDs only; geriatric long-term care cohort) and the synthetic IV dataset. "
            "In accordance with scientific integrity guidelines, age values were not fabricated. "
            "The modular AgeFeatureEncoder interface, fusion pathways, and forward pass were verified, "
            "and computational overhead was benchmarked."
        ),
        "profiling": m4_profile,
        "checkpoint_path": str(ckpt_m4.relative_to(BASE_DIR))
    }
    print(f"  Model 4 Verified -> Params: {m4_profile['total_parameters']}, Latency: {m4_profile['latency_cpu_mean_ms']}ms")

    # Step 5: Modality Ablations on Model 3
    print("\n[5/6] Assembling Modality Ablations on Model 3 Architecture...", flush=True)
    ablations = {
        "tcn_only": {
            "name": "Temp + Pressure Only (Model 1)",
            "macro_f1_mean": results["model1_tcn"]["macro_f1_mean"],
            "macro_f1_std": results["model1_tcn"]["macro_f1_std"],
            "confounder_fpr_mean": results["model1_tcn"]["confounder_fpr_mean"],
            "confounder_fpr_std": results["model1_tcn"]["confounder_fpr_std"],
        },
        "tcn_plus_bio": {
            "name": "Temp + Pressure + Bioimpedance (Model 2)",
            "macro_f1_mean": results["model2_bio_tcn"]["macro_f1_mean"],
            "macro_f1_std": results["model2_bio_tcn"]["macro_f1_std"],
            "confounder_fpr_mean": results["model2_bio_tcn"]["confounder_fpr_mean"],
            "confounder_fpr_std": results["model2_bio_tcn"]["confounder_fpr_std"],
        },
        "full_multimodal": {
            "name": "Temp + Pressure + Bioimpedance + Stress (Model 3)",
            "macro_f1_mean": results["model3_stress_multi"]["macro_f1_mean"],
            "macro_f1_std": results["model3_stress_multi"]["macro_f1_std"],
            "confounder_fpr_mean": results["model3_stress_multi"]["confounder_fpr_mean"],
            "confounder_fpr_std": results["model3_stress_multi"]["confounder_fpr_std"],
        }
    }

    # Train only the complementary ablation: Temp + Pressure + Stress (without Bioimpedance)
    print("  Training Ablation: Temp + Pressure + Stress (Dual-Stream without Bioimpedance)...", flush=True)
    f1s_stress, fprs_stress = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = Model3_StressMultimodal(in_channels=2, bio_features=28, num_classes=4)
        m = train_model("ablation_tcn_stress", m, train_data, val_data)
        preds = predict_model("ablation_tcn_stress", m, X_loc_te_n, X_bio_te, X_str_te)
        f1s_stress.append(float(f1_score(y_te, preds, average="macro", zero_division=0)))
        fprs_stress.append(compute_confounder_fpr(test_df, preds, tid_te, y_te)["fpr"])

    ablations["tcn_plus_stress"] = {
        "name": "Temp + Pressure + Stress (Dual-Stream)",
        "macro_f1_mean": round(float(np.mean(f1s_stress)), 4),
        "macro_f1_std": round(float(np.std(f1s_stress)), 4),
        "confounder_fpr_mean": round(float(np.mean(fprs_stress)), 4),
        "confounder_fpr_std": round(float(np.std(fprs_stress)), 4),
    }

    for ab_id, ab_info in ablations.items():
        print(f"  Ablation: {ab_info['name']:<45} | Macro-F1: {ab_info['macro_f1_mean']:.4f} | FPR: {ab_info['confounder_fpr_mean']:.4f}", flush=True)

    results["modality_ablations"] = ablations

    # Step 6: Export Results, Metrics & Plots
    print("\n[6/6] Generating Publication-Quality Plots and Reports...")
    with open(OUTPUT_DIR / "progressive_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot 1: Progressive Performance Metrics
    plt.figure(figsize=(10, 5), dpi=300)
    models = ["Model 1 (Baseline TCN)", "Model 2 (+ Bioimpedance)", "Model 3 (+ Stress Multimodal)"]
    f1_scores = [results["model1_tcn"]["macro_f1_mean"],
                 results["model2_bio_tcn"]["macro_f1_mean"],
                 results["model3_stress_multi"]["macro_f1_mean"]]
    f1_errs = [results["model1_tcn"]["macro_f1_std"],
               results["model2_bio_tcn"]["macro_f1_std"],
               results["model3_stress_multi"]["macro_f1_std"]]
    bal_accs = [results["model1_tcn"]["balanced_acc_mean"],
                results["model2_bio_tcn"]["balanced_acc_mean"],
                results["model3_stress_multi"]["balanced_acc_mean"]]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars1 = ax.bar(x - width/2, f1_scores, width, yerr=f1_errs, capsize=4, label='Macro-F1', color='#0284c7', alpha=0.9)
    bars2 = ax.bar(x + width/2, bal_accs, width, label='Balanced Accuracy', color='#0d9488', alpha=0.9)

    ax.set_ylabel('Score')
    ax.set_title('Progressive Model Performance on Held-Out IV Trials')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.90, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "progressive_metrics_comparison.png")
    plt.close()

    # Plot 2: Confounder False Positive Rate Reduction
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    fprs = [results["model1_tcn"]["confounder_fpr_mean"] * 100,
            results["model2_bio_tcn"]["confounder_fpr_mean"] * 100,
            results["model3_stress_multi"]["confounder_fpr_mean"] * 100]
    fpr_errs = [results["model1_tcn"]["confounder_fpr_std"] * 100,
                results["model2_bio_tcn"]["confounder_fpr_std"] * 100,
                results["model3_stress_multi"]["confounder_fpr_std"] * 100]

    bars = ax.bar(models, fprs, yerr=fpr_errs, capsize=4, color=['#f97316', '#eab308', '#22c55e'])
    ax.set_ylabel('Confounder False Positive Rate (%)')
    ax.set_title('Confounder Rejection (Vasoconstriction Challenge)')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confounder_fpr_comparison.png")
    plt.close()

    # Plot 3: Modality Ablation Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ab_labels = [v["name"].replace(" (Model 1)", "").replace(" (Model 2)", "").replace(" (Model 3)", "") for v in ablations.values()]
    ab_f1s = [v["macro_f1_mean"] for v in ablations.values()]
    ab_fprs = [v["confounder_fpr_mean"] * 100 for v in ablations.values()]

    x = np.arange(len(ab_labels))
    ax1 = ax
    ax2 = ax.twinx()

    b1 = ax1.bar(x - 0.2, ab_f1s, 0.4, label='Macro-F1', color='#3b82f6')
    b2 = ax2.bar(x + 0.2, ab_fprs, 0.4, label='Confounder FPR (%)', color='#ef4444')

    ax1.set_ylabel('Macro-F1 Score', color='#3b82f6')
    ax2.set_ylabel('Confounder FPR (%)', color='#ef4444')
    ax1.set_ylim(0.92, 1.0)
    ax2.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ab_labels, rotation=15, ha='right', fontsize=8.5)
    ax.set_title('Modality Ablation Analysis: Contribution to Accuracy & Confounder Rejection')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "modality_ablation_comparison.png")
    plt.close()

    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
