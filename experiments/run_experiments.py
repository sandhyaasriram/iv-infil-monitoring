"""
Master Experiment Runner:
1. Pre-trains Systemic Autonomic Encoder on Real WESAD multi-modal signals.
2. Trains Baselines: Logistic Regression, Random Forest, Shallow 1D-CNN, and Deep 1D-ResNet.
3. Trains Proposed Autonomic-Decoupled Dual-Stream Model.
4. Evaluates on Clean Held-out Test Trials.
5. Evaluates on Confounder Challenge (Real Human Vasoconstriction).
6. Runs Low-Data Regime Ablation (20% IV Data Scarcity Test).
7. Exports results to JSON and Markdown.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.splits import split_iv_trials, WESAD_TRAIN_SUBJECTS, WESAD_VAL_SUBJECTS, WESAD_TEST_SUBJECTS
from src.evaluation.metrics import (
    compute_classification_metrics, compute_confounder_fpr, compute_detection_delays, CLASS_NAMES
)
from src.data.wesad_loader import WESADWindowDataset
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


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pytorch_model(model, train_loader, val_loader, epochs=15, lr=1e-3, device='cpu', weight_decay=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Pre-train Systemic Encoder on Real WESAD Data
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 1: PRE-TRAINING SYSTEMIC ENCODER ON REAL WESAD")
    print("=======================================================")
    wesad_dir = r"c:\mer\iot projecttt\WESAD"
    print(f"Loading WESAD train subjects: {WESAD_TRAIN_SUBJECTS}")
    wesad_train_ds = WESADWindowDataset(wesad_dir, WESAD_TRAIN_SUBJECTS, window_size=100, step_size=50)
    wesad_val_ds = WESADWindowDataset(wesad_dir, WESAD_VAL_SUBJECTS, window_size=100, step_size=50, scaler_stats=wesad_train_ds.scaler_stats)
    wesad_test_ds = WESADWindowDataset(wesad_dir, WESAD_TEST_SUBJECTS, window_size=100, step_size=50, scaler_stats=wesad_train_ds.scaler_stats)

    wesad_train_loader = DataLoader(wesad_train_ds, batch_size=64, shuffle=True)
    wesad_val_loader = DataLoader(wesad_val_ds, batch_size=64, shuffle=False)
    wesad_test_loader = DataLoader(wesad_test_ds, batch_size=64, shuffle=False)

    print(f"WESAD Windows: Train={len(wesad_train_ds)}, Val={len(wesad_val_ds)}, Test={len(wesad_test_ds)}")

    systemic_encoder = SystemicAutonomicEncoder(in_channels=4, latent_dim=32).to(device)
    # Pre-train stress prediction head
    stress_crit = nn.CrossEntropyLoss()
    stress_opt = torch.optim.AdamW(systemic_encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    encoder_ckpt = os.path.join(results_dir, 'systemic_encoder_wesad.pt')
    if os.path.exists(encoder_ckpt):
        print(f"Loading existing pre-trained Systemic Encoder from {encoder_ckpt}...")
        systemic_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    else:
        print("Training Systemic Autonomic Encoder on real physiological signals (10 epochs)...")
        for ep in range(10):
            systemic_encoder.train()
            tr_loss = 0.0
            for x, y in wesad_train_loader:
                x, y = x.to(device), y.to(device)
                stress_opt.zero_grad()
                logits = systemic_encoder.predict_stress(x)
                loss = stress_crit(logits, y)
                loss.backward()
                stress_opt.step()
                tr_loss += loss.item()
            tr_loss /= len(wesad_train_loader)
            if (ep + 1) % 5 == 0:
                print(f"  Epoch {ep+1}/10 - WESAD Pre-train Loss: {tr_loss:.4f}")
        torch.save(systemic_encoder.state_dict(), encoder_ckpt)

    # Evaluate pre-trained encoder on held-out test subjects
    systemic_encoder.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in wesad_test_loader:
            x, y = x.to(device), y.to(device)
            preds = systemic_encoder.predict_stress(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
    wesad_acc = correct / max(1, total)
    print(f"WESAD Pre-training Accuracy on held-out test subjects (S16, S17): {wesad_acc*100:.2f}%")


    # ---------------------------------------------------------
    # Step 2: Load and Prepare IV Dataset (Clean + Confounders)
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 2: LOADING IV DATASET WITH STRICT TRIAL SPLIT")
    print("=======================================================")
    csv_path = "synthetic_iv_infiltration_dataset.csv"
    df_raw = pd.read_csv(csv_path)

    # Perform trial-disjoint splitting (No window leakage)
    splits = split_iv_trials(df_raw, test_size=0.25, val_size=0.15, random_state=42)
    train_df_clean = splits['train_df']
    val_df_clean = splits['val_df']
    test_df_clean = splits['test_df']

    print(f"Clean Trials: Train={len(splits['train_trials'])}, Val={len(splits['val_trials'])}, Test={len(splits['test_trials'])}")

    # Inject real human vasoconstriction confounders into test split for the Confounder Challenge
    print("Generating Confounder Challenge test set using real human vasoconstriction...")
    test_df_challenge = inject_systemic_confounders(test_df_clean, wesad_test_loader, n_confounder_trials=15, seed=42)

    # Extract Tabular features for ML baselines
    print("Extracting tabular window features for tree/linear baselines...")
    tab_train = extract_tabular_window_features(train_df_clean, window_s=25)
    tab_val = extract_tabular_window_features(val_df_clean, window_s=25)
    tab_test_clean = extract_tabular_window_features(test_df_clean, window_s=25)
    tab_test_challenge = extract_tabular_window_features(test_df_challenge, window_s=25)

    feat_cols = [c for c in tab_train.columns if c not in ['label', 'trial_id', 'condition', 'severity', 'time_s', 'is_confounder']]
    X_train_tab, y_train_tab = tab_train[feat_cols], tab_train['label']
    X_val_tab, y_val_tab = tab_val[feat_cols], tab_val['label']
    X_test_clean_tab, y_test_clean_tab = tab_test_clean[feat_cols], tab_test_clean['label']
    X_test_chal_tab, y_test_chal_tab = tab_test_challenge[feat_cols], tab_test_challenge['label']

    # PyTorch Temporal Datasets
    print("Building temporal tensor datasets for neural networks...")
    iv_train_ds = IVTemporalWindowDataset(train_df_clean, window_s=25)
    iv_val_ds = IVTemporalWindowDataset(val_df_clean, window_s=25, scaler_stats=iv_train_ds.scaler_stats)
    iv_test_clean_ds = IVTemporalWindowDataset(test_df_clean, window_s=25, scaler_stats=iv_train_ds.scaler_stats)
    iv_test_chal_ds = IVTemporalWindowDataset(test_df_challenge, window_s=25, scaler_stats=iv_train_ds.scaler_stats)

    iv_train_loader = DataLoader(iv_train_ds, batch_size=64, shuffle=True)
    iv_val_loader = DataLoader(iv_val_ds, batch_size=64, shuffle=False)
    iv_test_clean_loader = DataLoader(iv_test_clean_ds, batch_size=64, shuffle=False)
    iv_test_chal_loader = DataLoader(iv_test_chal_ds, batch_size=64, shuffle=False)

    # ---------------------------------------------------------
    # Step 3: Train All Baselines & Proposed Model
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 3: TRAINING BASELINES & PROPOSED DUAL-STREAM MODEL")
    print("=======================================================")

    models_to_evaluate = {}

    # 1. Baseline 1: Logistic Regression
    print("Fitting Baseline 1: Logistic Regression...")
    pipe_lr = build_logistic_regression_baseline(random_state=42)
    pipe_lr.fit(X_train_tab, y_train_tab)
    models_to_evaluate['Baseline_LogisticRegression'] = {
        'type': 'tabular', 'model': pipe_lr, 'feats': feat_cols
    }

    # 2. Baseline 2: Random Forest
    print("Fitting Baseline 2: Random Forest...")
    pipe_rf = build_random_forest_baseline(random_state=42)
    pipe_rf.fit(X_train_tab, y_train_tab)
    models_to_evaluate['Baseline_RandomForest'] = {
        'type': 'tabular', 'model': pipe_rf, 'feats': feat_cols
    }

    # 3. Baseline 3: Shallow 1D-CNN
    print("Training Baseline 3: Shallow 1D-CNN...")
    shallow_cnn = ShallowCNN1D(in_channels=2, num_classes=4)
    shallow_cnn = train_pytorch_model(shallow_cnn, iv_train_loader, iv_val_loader, epochs=15, lr=1e-3, device=device)
    models_to_evaluate['Baseline_ShallowCNN1D'] = {
        'type': 'pytorch', 'model': shallow_cnn
    }

    # 4. Baseline 4: Deep 1D-ResNet (NEW DEEPER BASELINE)
    print("Training Baseline 4: Deep 1D-ResNet (6 residual layers)...")
    deep_resnet = DeepResNet1D(in_channels=2, num_classes=4, base_channels=32)
    deep_resnet = train_pytorch_model(deep_resnet, iv_train_loader, iv_val_loader, epochs=15, lr=1e-3, device=device)
    models_to_evaluate['Baseline_DeepResNet1D'] = {
        'type': 'pytorch', 'model': deep_resnet
    }

    # 5. Proposed Model: Autonomic-Decoupled Dual-Stream Network
    print("Training Proposed Model: Autonomic-Decoupled Dual-Stream Network...")
    dual_stream = AutonomicDecoupledDualStreamModel(systemic_encoder=systemic_encoder, num_classes=4, latent_dim=32, freeze_systemic=True).to(device)
    crit_dual = nn.CrossEntropyLoss()
    opt_dual = torch.optim.AdamW(dual_stream.parameters(), lr=1e-3, weight_decay=1e-4)

    # Prepare paired systemic sample iterator for training
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

            x_sys = x_sys[:len(x_loc)].to(device)
            # If batch sizes differ, pad or slice
            if len(x_sys) < len(x_loc):
                x_sys = torch.cat([x_sys, x_sys[:len(x_loc)-len(x_sys)]], dim=0)

            opt_dual.zero_grad()
            logits = dual_stream(x_loc, x_sys)
            loss = crit_dual(logits, y_loc)
            loss.backward()
            opt_dual.step()

    models_to_evaluate['Proposed_DualStream'] = {
        'type': 'dual_stream', 'model': dual_stream
    }

    # ---------------------------------------------------------
    # Step 4: Comprehensive Evaluation (Clean & Confounder Challenge)
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 4: EVALUATING ON CLEAN & CONFOUNDER TEST SETS")
    print("=======================================================")

    experiment_results = {}

    for name, entry in models_to_evaluate.items():
        print(f"\n--- Evaluating {name} ---")
        m_type = entry['type']
        model = entry['model']

        # 1. Clean Test Predictions
        if m_type == 'tabular':
            y_pred_clean = model.predict(X_test_clean_tab)
            y_pred_chal = model.predict(X_test_chal_tab)
        elif m_type == 'pytorch':
            model.eval()
            preds_clean, preds_chal = [], []
            with torch.no_grad():
                for x, _, _, _, _ in iv_test_clean_loader:
                    preds_clean.extend(model(x.to(device)).argmax(dim=-1).cpu().numpy())
                for x, _, _, _, _ in iv_test_chal_loader:
                    preds_chal.extend(model(x.to(device)).argmax(dim=-1).cpu().numpy())
            y_pred_clean = np.array(preds_clean)
            y_pred_chal = np.array(preds_chal)
        elif m_type == 'dual_stream':
            model.eval()
            preds_clean, preds_chal = [], []
            # On clean: neutral systemic baseline
            with torch.no_grad():
                for x, _, _, _, _ in iv_test_clean_loader:
                    preds_clean.extend(model(x.to(device), None).argmax(dim=-1).cpu().numpy())
                # On confounder challenge: pass real systemic WESAD signals
                w_test_it = iter(wesad_test_loader)
                for x, _, _, _, _ in iv_test_chal_loader:
                    try:
                        x_s, _ = next(w_test_it)
                    except StopIteration:
                        w_test_it = iter(wesad_test_loader)
                        x_s, _ = next(w_test_it)
                    x_s = x_s[:len(x)].to(device)
                    if len(x_s) < len(x):
                        x_s = torch.cat([x_s, x_s[:len(x)-len(x_s)]], dim=0)
                    preds_chal.extend(model(x.to(device), x_s).argmax(dim=-1).cpu().numpy())
            y_pred_clean = np.array(preds_clean)
            y_pred_chal = np.array(preds_chal)

        # Compute Metrics
        metrics_clean = compute_classification_metrics(y_test_clean_tab, y_pred_clean)
        fpr_confounder = compute_confounder_fpr(
            y_test_chal_tab, y_pred_chal, tab_test_challenge['is_confounder'].values
        )

        # Detection Delay on clean test set
        df_delay = tab_test_clean.copy()
        df_delay['pred'] = y_pred_clean
        delay_stats = compute_detection_delays(df_delay, pred_col='pred', label_col='label')

        experiment_results[name] = {
            'clean_macro_f1': float(metrics_clean['macro_f1']),
            'clean_accuracy': float(metrics_clean['accuracy']),
            'early_recall': float(metrics_clean['per_class']['early']['recall']),
            'confounder_fpr': float(fpr_confounder),
            'mean_delay_s': float(delay_stats['mean_delay_s']),
            'median_delay_s': float(delay_stats['median_delay_s']),
            'per_class': metrics_clean['per_class'],
            'confusion_matrix': metrics_clean['confusion_matrix']
        }

        print(f"  Clean Macro-F1: {metrics_clean['macro_f1']:.4f}")
        print(f"  Early-Class Recall: {metrics_clean['per_class']['early']['recall']:.4f}")
        print(f"  Confounder False Positive Rate (FPR): {fpr_confounder*100:.2f}%")
        print(f"  Mean Detection Delay: {delay_stats['mean_delay_s']:.1f}s")

    # ---------------------------------------------------------
    # Step 5: Data Scarcity Ablation (Low-Data Regime: 20% IV Data)
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 5: DATA SCARCITY ABLATION (20% IV TRAINING DATA)")
    print("=======================================================")
    rng = np.random.default_rng(42)
    avail_train_trials = splits['train_trials']
    scarce_train_trials = rng.choice(avail_train_trials, size=max(4, int(len(avail_train_trials) * 0.2)), replace=False)

    tab_train_scarce = tab_train[tab_train['trial_id'].isin(scarce_train_trials)]
    X_train_scarce, y_train_scarce = tab_train_scarce[feat_cols], tab_train_scarce['label']

    iv_train_scarce_df = train_df_clean[train_df_clean['trial_id'].isin(scarce_train_trials)]
    iv_train_scarce_ds = IVTemporalWindowDataset(iv_train_scarce_df, window_s=25, scaler_stats=iv_train_ds.scaler_stats)
    iv_train_scarce_loader = DataLoader(iv_train_scarce_ds, batch_size=32, shuffle=True)

    scarce_results = {}

    # Scarcity RF
    rf_scarce = build_random_forest_baseline(random_state=42)
    rf_scarce.fit(X_train_scarce, y_train_scarce)
    f1_rf_scarce = compute_classification_metrics(y_test_clean_tab, rf_scarce.predict(X_test_clean_tab))['macro_f1']
    scarce_results['Baseline_RandomForest_20pct'] = float(f1_rf_scarce)

    # Scarcity Deep ResNet
    resnet_scarce = DeepResNet1D(in_channels=2, num_classes=4, base_channels=32)
    resnet_scarce = train_pytorch_model(resnet_scarce, iv_train_scarce_loader, iv_val_loader, epochs=15, lr=1e-3, device=device)
    resnet_scarce.eval()
    preds_sc = []
    with torch.no_grad():
        for x, _, _, _, _ in iv_test_clean_loader:
            preds_sc.extend(resnet_scarce(x.to(device)).argmax(dim=-1).cpu().numpy())
    f1_resnet_scarce = compute_classification_metrics(y_test_clean_tab, np.array(preds_sc))['macro_f1']
    scarce_results['Baseline_DeepResNet1D_20pct'] = float(f1_resnet_scarce)

    # Scarcity Proposed Dual-Stream
    dual_scarce = AutonomicDecoupledDualStreamModel(systemic_encoder=systemic_encoder, num_classes=4, latent_dim=32, freeze_systemic=True).to(device)
    opt_dual_sc = torch.optim.AdamW(dual_scarce.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(15):
        dual_scarce.train()
        for x_loc, y_loc, _, _, _ in iv_train_scarce_loader:
            x_loc, y_loc = x_loc.to(device), y_loc.to(device)
            try:
                x_sys, _ = next(wesad_iter)
            except StopIteration:
                wesad_iter = iter(wesad_train_loader)
                x_sys, _ = next(wesad_iter)
            x_sys = x_sys[:len(x_loc)].to(device)
            if len(x_sys) < len(x_loc):
                x_sys = torch.cat([x_sys, x_sys[:len(x_loc)-len(x_sys)]], dim=0)
            opt_dual_sc.zero_grad()
            loss = crit_dual(dual_scarce(x_loc, x_sys), y_loc)
            loss.backward()
            opt_dual_sc.step()

    dual_scarce.eval()
    preds_dual_sc = []
    with torch.no_grad():
        for x, _, _, _, _ in iv_test_clean_loader:
            preds_dual_sc.extend(dual_scarce(x.to(device), None).argmax(dim=-1).cpu().numpy())
    f1_dual_scarce = compute_classification_metrics(y_test_clean_tab, np.array(preds_dual_sc))['macro_f1']
    scarce_results['Proposed_DualStream_20pct'] = float(f1_dual_scarce)

    print(f"  Random Forest (20% IV Data) Macro-F1: {f1_rf_scarce:.4f}")
    print(f"  Deep ResNet-1D (20% IV Data) Macro-F1: {f1_resnet_scarce:.4f}")
    print(f"  Proposed Dual-Stream (20% IV Data) Macro-F1: {f1_dual_scarce:.4f}")

    # ---------------------------------------------------------
    # Step 6: Save Comprehensive Results Summary
    # ---------------------------------------------------------
    final_output = {
        'clean_evaluation': experiment_results,
        'scarcity_ablation': scarce_results,
        'wesad_pretrain_accuracy': float(wesad_acc)
    }

    with open(os.path.join(results_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(final_output, f, indent=2)

    print("\nExperiment summary saved to results/experiment_summary.json.")


if __name__ == '__main__':
    main()
