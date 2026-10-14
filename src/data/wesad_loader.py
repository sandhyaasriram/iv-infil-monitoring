"""
WESAD Dataset Loader
Synchronizes Empatica E4 multi-rate wrist sensors (TEMP 4Hz, EDA 4Hz, BVP 64Hz, ACC 32Hz)
to synchronous 4 Hz frames and extracts leak-free subject-wise windows.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

WESAD_TRAIN_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13']
WESAD_VAL_SUBJECTS = ['S14', 'S15']
WESAD_TEST_SUBJECTS = ['S16', 'S17']


def load_subject_wrist_4hz(wesad_dir, subject_id):
    """
    Loads Empatica E4 wrist signals and aggregates them synchronously at 4 Hz.
    Channels:
    0: TEMP (deg C)
    1: EDA (uS)
    2: BVP_AMP (pulse envelope variability)
    3: ACC_MAG (motion magnitude, g)
    """
    pkl_path = os.path.join(wesad_dir, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"WESAD subject file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    wrist = data['signal']['wrist']
    raw_labels = data['label'] # 700 Hz ground-truth protocol labels

    temp_4hz = wrist['TEMP'].flatten()
    eda_4hz = wrist['EDA'].flatten()
    n_samples_4hz = min(len(temp_4hz), len(eda_4hz))

    # BVP (64 Hz) -> 4 Hz (16 samples per step)
    bvp_raw = wrist['BVP'].flatten()[:n_samples_4hz * 16]
    bvp_blocks = bvp_raw.reshape(n_samples_4hz, 16)
    bvp_amp_4hz = np.std(bvp_blocks, axis=1) # pulse volume amplitude

    # ACC (32 Hz) -> 4 Hz (8 samples per step)
    acc_raw = wrist['ACC'][:n_samples_4hz * 8]
    acc_mag = np.linalg.norm(acc_raw, axis=1)
    acc_mag_blocks = acc_mag.reshape(n_samples_4hz, 8)
    acc_mag_4hz = np.mean(acc_mag_blocks, axis=1)

    temp_4hz = temp_4hz[:n_samples_4hz]
    eda_4hz = eda_4hz[:n_samples_4hz]

    # Labels: 700 Hz -> 4 Hz (175 samples per step)
    label_samples = min(len(raw_labels), n_samples_4hz * 175)
    labels_reshaped = raw_labels[:label_samples].reshape(-1, 175)
    labels_4hz = np.array([np.bincount(row).argmax() for row in labels_reshaped])
    if len(labels_4hz) < n_samples_4hz:
        labels_4hz = np.pad(labels_4hz, (0, n_samples_4hz - len(labels_4hz)), mode='edge')
    else:
        labels_4hz = labels_4hz[:n_samples_4hz]

    X_4hz = np.stack([temp_4hz, eda_4hz, bvp_amp_4hz, acc_mag_4hz], axis=1)
    return X_4hz, labels_4hz


class WESADWindowDataset(Dataset):
    """
    Extracts sliding windows of length window_size (default 100 samples = 25s at 4 Hz)
    for training the systemic autonomic encoder.
    """
    def __init__(self, wesad_dir, subjects, window_size=100, step_size=50, standardize=True, scaler_stats=None):
        self.windows = []
        self.labels = []
        self.subject_ids = []

        all_signals = []
        for s_id in subjects:
            X, y = load_subject_wrist_4hz(wesad_dir, s_id)
            all_signals.append((s_id, X, y))

        if standardize:
            if scaler_stats is None:
                all_X = np.concatenate([item[1] for item in all_signals], axis=0)
                mean = np.nanmean(all_X, axis=0)
                std = np.nanstd(all_X, axis=0) + 1e-6
                self.scaler_stats = {'mean': mean, 'std': std}
            else:
                self.scaler_stats = scaler_stats
        else:
            self.scaler_stats = None

        for s_id, X, y in all_signals:
            if self.scaler_stats is not None:
                X = (X - self.scaler_stats['mean']) / self.scaler_stats['std']

            n = len(X)
            for start in range(0, n - window_size, step_size):
                w_x = X[start:start+window_size].T # shape (4, window_size)
                w_y = np.bincount(y[start:start+window_size]).argmax()
                # 1 = Sympathetic Stress (vasoconstriction), 0 = Non-stress baseline
                binary_stress = 1 if w_y == 1 else 0

                self.windows.append(w_x)
                self.labels.append(binary_stress)
                self.subject_ids.append(s_id)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.labels[idx])
