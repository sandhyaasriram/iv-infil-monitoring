# Metrics Evaluation Module
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

CLASS_NAMES = ['none', 'early', 'moderate', 'severe']

def compute_classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    per_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        p = precision_score(y_true == c_idx, y_pred == c_idx, zero_division=0)
        r = recall_score(y_true == c_idx, y_pred == c_idx, zero_division=0)
        f = f1_score(y_true == c_idx, y_pred == c_idx, zero_division=0)
        per_class[c_name] = {'precision': p, 'recall': r, 'f1': f}
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': acc,
        'per_class': per_class,
        'confusion_matrix': cm.tolist()
    }

def compute_confounder_fpr(y_true, y_pred, is_confounder):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_confounder = np.asarray(is_confounder, dtype=bool)
    
    # We evaluate on windows where true status is NONE (no fluid leaked) during confounder episodes
    neg_confounder_mask = (y_true == 0) & is_confounder
    if np.sum(neg_confounder_mask) == 0:
        return 0.0
    
    # False positive = model predicts class >= 1
    false_alarms = np.sum((y_pred >= 1) & neg_confounder_mask)
    total_negs = np.sum(neg_confounder_mask)
    return float(false_alarms / total_negs)

def compute_detection_delays(df_eval, pred_col='pred', label_col='label', time_col='time_s', trial_col='trial_id', min_consecutive=2):
    delays = []
    for t_id, grp in df_eval.groupby(trial_col):
        grp = grp.sort_values(time_col)
        y_true = grp[label_col].values
        y_pred = grp[pred_col].values
        times = grp[time_col].values
        
        # Check if trial has true infiltration
        true_onset_idx = np.where(y_true >= 1)[0]
        if len(true_onset_idx) == 0:
            continue
        t_onset = times[true_onset_idx[0]]
        
        # Find first sustained detection
        detected_idx = None
        for i in range(len(y_pred) - min_consecutive + 1):
            if np.all(y_pred[i:i+min_consecutive] >= 1):
                detected_idx = i
                break
                
        if detected_idx is not None:
            t_det = times[detected_idx]
            delays.append(max(0.0, t_det - t_onset))
        else:
            # Did not detect before trial end: penalty equal to remaining trial duration
            delays.append(times[-1] - t_onset)
            
    return {
        'mean_delay_s': float(np.mean(delays)) if delays else 0.0,
        'median_delay_s': float(np.median(delays)) if delays else 0.0,
        'num_trials_evaluated': len(delays)
    }
