"""
Comprehensive Technical Audit Script
Performs in-depth verification of:
1. Data splits & leakage
2. Confounder generation reality vs claims
3. Detection delay per-trial analysis & negative delays
4. Confounder false-alarm denominators & confidence intervals
5. Multi-seed data scarcity ablation (mean +/- std)
6. Dual-Stream ablation (with vs without systemic inputs)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.splits import split_iv_trials, WESAD_TRAIN_SUBJECTS, WESAD_VAL_SUBJECTS, WESAD_TEST_SUBJECTS
from src.evaluation.metrics import (
    compute_classification_metrics, compute_confounder_fpr, compute_detection_delays, CLASS_NAMES
)
from src.data.wesad_loader import WESADWindowDataset, load_subject_wrist_4hz
from src.data.iv_dataset import (
    extract_tabular_window_features, IVTemporalWindowDataset, inject_systemic_confounders
)
from src.models.baselines import (
    build_logistic_regression_baseline, build_random_forest_baseline,
    ShallowCNN1D, DeepResNet1D
)
from src.models.dual_stream import (
    SystemicAutonomicEncoder, AutonomicDecoupledDualStreamModel
)


def run_full_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("STARTING TECHNICAL AUDIT EXECUTION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Audit WESAD Dataset & Splits
    # -------------------------------------------------------------
    wesad_dir = r"c:\mer\iot projecttt\WESAD"
    actual_subjs = sorted([s for s in os.listdir(wesad_dir) if s.startswith('S') and os.path.isdir(os.path.join(wesad_dir, s))])
    print(f"Actual WESAD Subjects on disk ({len(actual_subjs)}): {actual_subjs}")

    configured_subjs = WESAD_TRAIN_SUBJECTS + WESAD_VAL_SUBJECTS + WESAD_TEST_SUBJECTS
    print(f"Configured in code ({len(configured_subjs)}): {configured_subjs}")
    assert set(actual_subjs) == set(configured_subjs), "Mismatch between disk and configured WESAD subjects!"
    
    # Check disjointness
    set_train = set(WESAD_TRAIN_SUBJECTS)
    set_val = set(WESAD_VAL_SUBJECTS)
    set_test = set(WESAD_TEST_SUBJECTS)
    assert len(set_train.intersection(set_val)) == 0, "Train-Val overlap in WESAD!"
    assert len(set_train.intersection(set_test)) == 0, "Train-Test overlap in WESAD!"
    assert len(set_val.intersection(set_test)) == 0, "Val-Test overlap in WESAD!"
    print("[PASS] WESAD subject splits are strictly disjoint (0 subject leakage).")

    # -------------------------------------------------------------
    # 2. Audit IV Dataset & Trial Splits
    # -------------------------------------------------------------
    df_raw = pd.read_csv("synthetic_iv_infiltration_dataset.csv")
    splits = split_iv_trials(df_raw, test_size=0.25, val_size=0.15, random_state=42)
    train_trials = set(splits['train_trials'])
    val_trials = set(splits['val_trials'])
    test_trials = set(splits['test_trials'])
    assert len(train_trials.intersection(val_trials)) == 0
    assert len(train_trials.intersection(test_trials)) == 0
    assert len(val_trials.intersection(test_trials)) == 0
    print(f"[PASS] IV trial splits are strictly disjoint: Train={len(train_trials)}, Val={len(val_trials)}, Test={len(test_trials)}.")

    # -------------------------------------------------------------
    # 3. Audit Confounder Generation Mechanism
    # -------------------------------------------------------------
    print("\n--- Auditing Confounder Injection Code ---")
    # Let's inspect inject_systemic_confounders
    # We load wesad test dataset
    wesad_train_ds = WESADWindowDataset(wesad_dir, WESAD_TRAIN_SUBJECTS, window_size=100, step_size=50)
    wesad_test_ds = WESADWindowDataset(wesad_dir, WESAD_TEST_SUBJECTS, window_size=100, step_size=50, scaler_stats=wesad_train_ds.scaler_stats)
    wesad_test_loader = DataLoader(wesad_test_ds, batch_size=64, shuffle=False)

    test_df_clean = splits['test_df'].copy()
    test_df_challenge = inject_systemic_confounders(test_df_clean, wesad_test_loader, n_confounder_trials=15, seed=42)
    conf_trials = test_df_challenge[test_df_challenge['is_confounder'] == 1]
    print(f"Confounder rows generated: {len(conf_trials)} across {conf_trials['trial_id'].nunique()} trials.")
    
    # Check what features differ between clean baseline and confounder in tab features
    tab_clean = extract_tabular_window_features(test_df_clean, window_s=25)
    tab_chal = extract_tabular_window_features(test_df_challenge, window_s=25)
    
    tab_conf_only = tab_chal[tab_chal['is_confounder'] == 1]
    print(f"Tabular Confounder windows: {len(tab_conf_only)}")
    print(f"Confounder pressure_adc mean: {tab_conf_only['press_mean'].mean():.2f} (std: {tab_conf_only['press_mean'].std():.2f})")
    print(f"Confounder pressure_adc slope: {tab_conf_only['press_slope'].mean():.4f} (std: {tab_conf_only['press_slope'].std():.4f})")
    print(f"Confounder temp_c mean: {tab_conf_only['temp_mean'].mean():.2f}")
    print(f"Confounder temp_c slope: {tab_conf_only['temp_slope'].mean():.4f}")

    # -------------------------------------------------------------
    # 4. Train Models & Audit Predictions
    # -------------------------------------------------------------
    print("\n--- Training Models for Exact Audit ---")
    feat_cols = [c for c in tab_clean.columns if c not in ['label', 'trial_id', 'condition', 'severity', 'time_s', 'is_confounder']]
    
    tab_train = extract_tabular_window_features(splits['train_df'], window_s=25)
    tab_val = extract_tabular_window_features(splits['val_df'], window_s=25)
    X_train_tab, y_train_tab = tab_train[feat_cols], tab_train['label']
    X_val_tab, y_val_tab = tab_val[feat_cols], tab_val['label']
    X_test_clean_tab, y_test_clean_tab = tab_clean[feat_cols], tab_clean['label']
    X_test_chal_tab, y_test_chal_tab = tab_chal[feat_cols], tab_chal['label']

    # Baseline 1: LogReg
    pipe_lr = build_logistic_regression_baseline(random_state=42)
    pipe_lr.fit(X_train_tab, y_train_tab)
    preds_lr_clean = pipe_lr.predict(X_test_clean_tab)
    preds_lr_chal = pipe_lr.predict(X_test_chal_tab)

    # Baseline 2: RF
    pipe_rf = build_random_forest_baseline(random_state=42)
    pipe_rf.fit(X_train_tab, y_train_tab)
    preds_rf_clean = pipe_rf.predict(X_test_clean_tab)
    preds_rf_chal = pipe_rf.predict(X_test_chal_tab)

    # PyTorch loaders
    iv_train_ds = IVTemporalWindowDataset(splits['train_df'], window_s=25)
    iv_val_ds = IVTemporalWindowDataset(splits['val_df'], window_s=25, scaler_stats=iv_train_ds.scaler_stats)
    iv_test_clean_ds = IVTemporalWindowDataset(test_df_clean, window_s=25, scaler_stats=iv_train_ds.scaler_stats)
    iv_test_chal_ds = IVTemporalWindowDataset(test_df_challenge, window_s=25, scaler_stats=iv_train_ds.scaler_stats)

    iv_train_loader = DataLoader(iv_train_ds, batch_size=64, shuffle=True)
    iv_val_loader = DataLoader(iv_val_ds, batch_size=64, shuffle=False)
    iv_test_clean_loader = DataLoader(iv_test_clean_ds, batch_size=64, shuffle=False)
    iv_test_chal_loader = DataLoader(iv_test_chal_ds, batch_size=64, shuffle=False)

    # Load pre-trained systemic encoder
    systemic_encoder = SystemicAutonomicEncoder(in_channels=4, latent_dim=32).to(device)
    encoder_ckpt = os.path.join(r"c:\mer\iot projecttt\iv-infiltration-project\results", 'systemic_encoder_wesad.pt')
    systemic_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))

    # Baseline 3: Shallow CNN
    def train_m(model, epochs=15):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            for x, y, _, _, _ in iv_train_loader:
                opt.zero_grad()
                loss = crit(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
        return model

    torch.manual_seed(42)
    shallow_cnn = train_m(ShallowCNN1D(in_channels=2, num_classes=4).to(device))
    shallow_cnn.eval()
    preds_cnn_clean, preds_cnn_chal = [], []
    with torch.no_grad():
        for x, _, _, _, _ in iv_test_clean_loader:
            preds_cnn_clean.extend(shallow_cnn(x.to(device)).argmax(dim=-1).cpu().numpy())
        for x, _, _, _, _ in iv_test_chal_loader:
            preds_cnn_chal.extend(shallow_cnn(x.to(device)).argmax(dim=-1).cpu().numpy())
    preds_cnn_clean = np.array(preds_cnn_clean)
    preds_cnn_chal = np.array(preds_cnn_chal)

    # Baseline 4: Deep ResNet
    torch.manual_seed(42)
    deep_resnet = train_m(DeepResNet1D(in_channels=2, num_classes=4, base_channels=32).to(device))
    deep_resnet.eval()
    preds_resnet_clean, preds_resnet_chal = [], []
    with torch.no_grad():
        for x, _, _, _, _ in iv_test_clean_loader:
            preds_resnet_clean.extend(deep_resnet(x.to(device)).argmax(dim=-1).cpu().numpy())
        for x, _, _, _, _ in iv_test_chal_loader:
            preds_resnet_chal.extend(deep_resnet(x.to(device)).argmax(dim=-1).cpu().numpy())
    preds_resnet_clean = np.array(preds_resnet_clean)
    preds_resnet_chal = np.array(preds_resnet_chal)

    # Proposed Dual Stream
    torch.manual_seed(42)
    dual_stream = AutonomicDecoupledDualStreamModel(systemic_encoder=systemic_encoder, num_classes=4, latent_dim=32, freeze_systemic=True).to(device)
    crit_dual = nn.CrossEntropyLoss()
    opt_dual = torch.optim.AdamW(dual_stream.parameters(), lr=1e-3, weight_decay=1e-4)
    wesad_train_loader = DataLoader(wesad_train_ds, batch_size=64, shuffle=True)
    wesad_iter = iter(wesad_train_loader)
    for ep in range(15):
        dual_stream.train()
        for x_loc, y_loc, _, _, _ in iv_train_loader:
            x_loc, y_loc = x_loc.to(device), y_loc.to(device)
            try:
                x_sys, _ = next(wesad_iter)
            except StopIteration:
                wesad_iter = iter(wesad_train_loader)
                x_sys, _ = next(wesad_iter)
            opt_dual.zero_grad()
            loss = crit_dual(dual_stream(x_loc, x_sys.to(device)), y_loc)
            loss.backward()
            opt_dual.step()

    dual_stream.eval()
    preds_dual_clean, preds_dual_chal = [], []
    preds_dual_chal_nosys = [] # ABLATION: without systemic input!
    with torch.no_grad():
        for x, _, _, _, _ in iv_test_clean_loader:
            preds_dual_clean.extend(dual_stream(x.to(device), None).argmax(dim=-1).cpu().numpy())
        
        w_test_it = iter(wesad_test_loader)
        for x, _, _, _, _ in iv_test_chal_loader:
            try:
                x_s, _ = next(w_test_it)
            except StopIteration:
                w_test_it = iter(wesad_test_loader)
                x_s, _ = next(w_test_it)
            preds_dual_chal.extend(dual_stream(x.to(device), x_s.to(device)).argmax(dim=-1).cpu().numpy())
            # ABLATION: evaluate with None as systemic input
            preds_dual_chal_nosys.extend(dual_stream(x.to(device), None).argmax(dim=-1).cpu().numpy())

    preds_dual_clean = np.array(preds_dual_clean)
    preds_dual_chal = np.array(preds_dual_chal)
    preds_dual_chal_nosys = np.array(preds_dual_chal_nosys)

    # -------------------------------------------------------------
    # 5. Detailed Detection Delay Audit
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AUDIT SECTION C: DETECTION DELAY DETAILED RECALCULATION")
    print("=" * 60)

    # Let's inspect every held-out trial in test_df_clean
    trial_info = []
    for t_id, grp in test_df_clean.groupby('trial_id'):
        sev = grp['severity'].iloc[0]
        leak_rate = grp['_leak_rate_mlph_GT'].iloc[0]
        # find true physical onset: where leak_rate > 0 first appears
        # in scenarios.py: ql_array[onset_idx:] = final_rate
        # let's check leaked_volume_ml_GT
        vols = grp['_leaked_volume_ml_GT'].values
        times = grp['time_s'].values
        leak_start_idx = np.where(vols > 0.0)[0]
        t_physical_leak_start = times[leak_start_idx[0]] if len(leak_start_idx) > 0 else np.nan
        
        # label >= 1 threshold (where volume >= 0.3 mL)
        labels = grp['label'].values
        label_1_idx = np.where(labels >= 1)[0]
        t_label_1_start = times[label_1_idx[0]] if len(label_1_idx) > 0 else np.nan

        trial_info.append({
            'trial_id': t_id,
            'severity': sev,
            'leak_rate_mlph': leak_rate,
            't_physical_leak_start': t_physical_leak_start,
            't_label_1_start': t_label_1_start
        })
    df_trial_info = pd.DataFrame(trial_info)
    print(f"Total held-out test trials: {len(df_trial_info)}")
    print(f"Trials by severity: {df_trial_info['severity'].value_counts().to_dict()}")

    # Recalculate detection delays trial-by-trial for ALL models
    all_models_preds = {
        'LogisticRegression': preds_lr_clean,
        'RandomForest': preds_rf_clean,
        'ShallowCNN1D': preds_cnn_clean,
        'DeepResNet1D': preds_resnet_clean,
        'DualStream': preds_dual_clean
    }

    delay_audit_results = {}

    for m_name, preds in all_models_preds.items():
        df_eval = tab_clean.copy()
        df_eval['pred'] = preds
        
        delays_from_label = []
        delays_from_physical = []
        negative_delay_count = 0
        detected_count = 0
        missed_count = 0

        # We evaluate on trials where severity != 'none'
        non_zero_trials = df_trial_info[df_trial_info['severity'] != 'none']
        
        for _, row in non_zero_trials.iterrows():
            t_id = row['trial_id']
            t_phys = row['t_physical_leak_start']
            t_lab1 = row['t_label_1_start']

            t_grp = df_eval[df_eval['trial_id'] == t_id].sort_values('time_s')
            p_arr = t_grp['pred'].values
            t_arr = t_grp['time_s'].values

            # Sustained prediction rule: min_consecutive = 2
            det_idx = None
            for i in range(len(p_arr) - 1):
                if p_arr[i] >= 1 and p_arr[i+1] >= 1:
                    det_idx = i
                    break

            if det_idx is not None:
                detected_count += 1
                t_det = t_arr[det_idx]
                d_lab = t_det - t_lab1
                d_phys = t_det - t_phys

                if d_lab < 0:
                    negative_delay_count += 1
                delays_from_label.append(d_lab)
                delays_from_physical.append(d_phys)
            else:
                missed_count += 1
                # Penalty: remaining duration from onset
                delays_from_label.append(t_arr[-1] - t_lab1)
                delays_from_physical.append(t_arr[-1] - t_phys)

        delay_audit_results[m_name] = {
            'detected': detected_count,
            'missed': missed_count,
            'negative_delays_count': negative_delay_count,
            'raw_delays_from_label': delays_from_label,
            'mean_raw_delay_from_label': float(np.mean(delays_from_label)),
            'median_raw_delay_from_label': float(np.median(delays_from_label)),
            'mean_clamped_delay_from_label': float(np.mean([max(0.0, d) for d in delays_from_label])),
            'mean_delay_from_physical': float(np.mean(delays_from_physical)),
            'median_delay_from_physical': float(np.median(delays_from_physical))
        }

    print("\n--- DELAY AUDIT TABLE ---")
    summary_delay = []
    for m_name, res in delay_audit_results.items():
        summary_delay.append({
            'Model': m_name,
            'Detected': f"{res['detected']}/{len(non_zero_trials)}",
            'Missed': res['missed'],
            'Negative Delays': res['negative_delays_count'],
            'Mean Raw Delay (vs Label=1)': f"{res['mean_raw_delay_from_label']:.1f}s",
            'Mean Clamped (max(0,d))': f"{res['mean_clamped_delay_from_label']:.1f}s",
            'Mean Delay (vs Phys Onset)': f"{res['mean_delay_from_physical']:.1f}s"
        })
    df_sum_delay = pd.DataFrame(summary_delay)
    print(df_sum_delay.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Detailed Confounder False Alarm Audit
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AUDIT SECTION D: CONFOUNDER FALSE ALARMS & CI")
    print("=" * 60)

    # In tab_chal, filter for windows where true label == 0 and is_confounder == 1
    conf_mask = (tab_chal['label'] == 0) & (tab_chal['is_confounder'] == 1)
    N_conf_windows = conf_mask.sum()
    N_conf_trials = tab_chal[conf_mask]['trial_id'].nunique()
    print(f"Total Confounder Evaluation Windows: {N_conf_windows} across {N_conf_trials} trials.")

    conf_audit_results = {}
    all_chal_preds = {
        'LogisticRegression': preds_lr_chal,
        'RandomForest': preds_rf_chal,
        'ShallowCNN1D': preds_cnn_chal,
        'DeepResNet1D': preds_resnet_chal,
        'DualStream_with_sys': preds_dual_chal,
        'DualStream_WITHOUT_sys': preds_dual_chal_nosys
    }

    for m_name, preds in all_chal_preds.items():
        false_alarms = np.sum((preds >= 1) & conf_mask)
        fpr = false_alarms / N_conf_windows
        # Wilson score interval for binomial proportion
        ci_low, ci_high = stats.norm.interval(0.95, loc=fpr, scale=np.sqrt(fpr * (1 - fpr) / N_conf_windows) if fpr > 0 else 0)
        ci_low = max(0.0, ci_low) if not np.isnan(ci_low) else 0.0
        ci_high = max(0.0, ci_high) if not np.isnan(ci_high) else 0.0

        # Trial-level false alarm: proportion of confounder trials with AT LEAST ONE false alarm
        df_eval_c = tab_chal.copy()
        df_eval_c['pred'] = preds
        trial_false_alarms = 0
        for t_id, grp in df_eval_c[df_eval_c['is_confounder'] == 1].groupby('trial_id'):
            if np.any(grp['pred'] >= 1):
                trial_false_alarms += 1
        trial_fpr = trial_false_alarms / N_conf_trials

        conf_audit_results[m_name] = {
            'window_false_alarms': int(false_alarms),
            'total_windows': int(N_conf_windows),
            'window_fpr': float(fpr),
            'window_fpr_95ci': (float(ci_low), float(ci_high)),
            'trial_false_alarms': int(trial_false_alarms),
            'total_trials': int(N_conf_trials),
            'trial_fpr': float(trial_fpr)
        }
        print(f"  {m_name}: Windows False Alarms={false_alarms}/{N_conf_windows} ({fpr*100:.2f}%), Trial FPR={trial_false_alarms}/{N_conf_trials} ({trial_fpr*100:.2f}%)")

    # -------------------------------------------------------------
    # 7. Multi-Seed Scarcity Ablation (Mean +/- Std)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AUDIT SECTION D.2: MULTI-SEED DATA SCARCITY (5 SEEDS)")
    print("=" * 60)

    seeds = [42, 101, 202, 303, 404]
    scarcity_table = {
        'RandomForest_20pct': [],
        'DeepResNet1D_20pct': [],
        'DualStream_20pct': []
    }

    avail_train_trials = splits['train_trials']

    for seed in seeds:
        print(f"Running Scarcity Ablation with Seed {seed}...")
        rng = np.random.default_rng(seed)
        scarce_trials = rng.choice(avail_train_trials, size=max(4, int(len(avail_train_trials) * 0.2)), replace=False)

        # Tabular scarce train
        tab_train_sc = tab_train[tab_train['trial_id'].isin(scarce_trials)]
        X_tr_sc, y_tr_sc = tab_train_sc[feat_cols], tab_train_sc['label']

        # RF
        rf_sc = build_random_forest_baseline(random_state=seed)
        rf_sc.fit(X_tr_sc, y_tr_sc)
        f1_rf = compute_classification_metrics(y_test_clean_tab, rf_sc.predict(X_test_clean_tab))['macro_f1']
        scarcity_table['RandomForest_20pct'].append(float(f1_rf))

        # Temporal scarce train
        iv_tr_sc_df = splits['train_df'][splits['train_df']['trial_id'].isin(scarce_trials)]
        iv_tr_sc_ds = IVTemporalWindowDataset(iv_tr_sc_df, window_s=25, scaler_stats=iv_train_ds.scaler_stats)
        iv_tr_sc_loader = DataLoader(iv_tr_sc_ds, batch_size=32, shuffle=True)

        # Deep ResNet
        torch.manual_seed(seed)
        res_sc = DeepResNet1D(in_channels=2, num_classes=4, base_channels=32).to(device)
        res_opt = torch.optim.AdamW(res_sc.parameters(), lr=1e-3, weight_decay=1e-4)
        res_crit = nn.CrossEntropyLoss()
        for ep in range(15):
            res_sc.train()
            for x_b, y_b, _, _, _ in iv_tr_sc_loader:
                res_opt.zero_grad()
                loss = res_crit(res_sc(x_b.to(device)), y_b.to(device))
                loss.backward()
                res_opt.step()
        res_sc.eval()
        p_sc = []
        with torch.no_grad():
            for x_b, _, _, _, _ in iv_test_clean_loader:
                p_sc.extend(res_sc(x_b.to(device)).argmax(dim=-1).cpu().numpy())
        f1_res = compute_classification_metrics(y_test_clean_tab, np.array(p_sc))['macro_f1']
        scarcity_table['DeepResNet1D_20pct'].append(float(f1_res))

        # Dual Stream
        torch.manual_seed(seed)
        dual_sc = AutonomicDecoupledDualStreamModel(systemic_encoder=systemic_encoder, num_classes=4, latent_dim=32, freeze_systemic=True).to(device)
        dual_opt = torch.optim.AdamW(dual_sc.parameters(), lr=1e-3, weight_decay=1e-4)
        dual_crit = nn.CrossEntropyLoss()
        w_iter_sc = iter(wesad_train_loader)
        for ep in range(15):
            dual_sc.train()
            for x_loc, y_loc, _, _, _ in iv_tr_sc_loader:
                try:
                    x_sys, _ = next(w_iter_sc)
                except StopIteration:
                    w_iter_sc = iter(wesad_train_loader)
                    x_sys, _ = next(w_iter_sc)
                dual_opt.zero_grad()
                loss = dual_crit(dual_sc(x_loc.to(device), x_sys.to(device)), y_loc.to(device))
                loss.backward()
                dual_opt.step()
        dual_sc.eval()
        p_dual_sc = []
        with torch.no_grad():
            for x_b, _, _, _, _ in iv_test_clean_loader:
                p_dual_sc.extend(dual_sc(x_b.to(device), None).argmax(dim=-1).cpu().numpy())
        f1_dual = compute_classification_metrics(y_test_clean_tab, np.array(p_dual_sc))['macro_f1']
        scarcity_table['DualStream_20pct'].append(float(f1_dual))

    print("\n--- MULTI-SEED SCARCITY RESULTS ---")
    for k, vals in scarcity_table.items():
        print(f"  {k}: Mean={np.mean(vals):.4f}, Std={np.std(vals):.4f}, Values={[round(v, 4) for v in vals]}")

    # Save complete audit dictionary
    audit_data = {
        'wesad_subjects': actual_subjs,
        'trial_split_counts': {'train': len(train_trials), 'val': len(val_trials), 'test': len(test_trials)},
        'delay_audit': delay_audit_results,
        'confounder_audit': conf_audit_results,
        'scarcity_multi_seed': scarcity_table
    }

    with open(r"c:\mer\iot projecttt\iv-infiltration-project\results\audit_reproducibility.json", 'w') as f:
        # Convert any numpy types to native python
        json.dump(audit_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else int(x) if isinstance(x, (np.int32, np.int64)) else x)
    print("\nFull audit data saved to results/audit_reproducibility.json.")


if __name__ == '__main__':
    run_full_audit()
