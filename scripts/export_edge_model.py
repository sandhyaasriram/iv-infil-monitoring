import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

WINDOW_S = 25

def extract_window_features(group):
    feats, labels, trial_ids, conditions, times = [], [], [], [], []
    temp = group['temperature_c'].values
    press = group['pressure_adc'].values
    lab = group['label'].values
    t_vals = group['time_s'].values
    n = len(group)

    for start in range(0, n - WINDOW_S + 1, 1): # slide by 1 to get continuous predictions
        t_win = temp[start:start+WINDOW_S]
        p_win = press[start:start+WINDOW_S]
        l_win = lab[start:start+WINDOW_S]
        
        # We assign the time of the *end* of the window
        curr_time = t_vals[start+WINDOW_S-1]

        def safe_stats(x, prefix):
            if np.all(np.isnan(x)):
                return {f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan,
                        f'{prefix}_slope': np.nan, f'{prefix}_min': np.nan,
                        f'{prefix}_max': np.nan}
            
            x_valid = np.nan_to_num(x, nan=np.nanmean(x))
            mean_val = np.nanmean(x)
            
            return {
                f'{prefix}_mean': mean_val,
                f'{prefix}_std': np.nanstd(x),
                f'{prefix}_slope': np.polyfit(np.arange(len(x_valid)), x_valid, 1)[0],
                f'{prefix}_min': np.nanmin(x),
                f'{prefix}_max': np.nanmax(x),
            }

        row = {}
        row.update(safe_stats(t_win, 'temp'))
        row.update(safe_stats(p_win, 'press'))
        row['sensor_missing_temp'] = int(np.all(np.isnan(t_win)))
        row['sensor_missing_press'] = int(np.all(np.isnan(p_win)))

        feats.append(row)
        labels.append(np.bincount(l_win).argmax())
        trial_ids.append(group['trial_id'].iloc[0])
        conditions.append(group['condition'].iloc[0])
        times.append(curr_time)

    out = pd.DataFrame(feats)
    out['label'] = labels
    out['trial_id'] = trial_ids
    out['condition'] = conditions
    out['time_s'] = times
    return out

print("Loading dataset...")
df = pd.read_csv("synthetic_iv_infiltration_dataset.csv")

print("Extracting features (this may take a minute)...")
feature_dfs = [extract_window_features(g) for _, g in df.groupby('trial_id')]
features = pd.concat(feature_dfs, ignore_index=True)

unique_trials = features['trial_id'].unique()
train_trials, test_trials = train_test_split(unique_trials, test_size=0.25, random_state=42)

train_df = features[features['trial_id'].isin(train_trials)]
test_df = features[features['trial_id'].isin(test_trials)]

feature_cols = [c for c in features.columns if c not in ['label', 'trial_id', 'condition', 'time_s']]

X_train, y_train = train_df[feature_cols], train_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

print("Training Logistic Regression...")
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
clf_logreg = LogisticRegression(max_iter=1000)

pipe_logreg = Pipeline([
    ('impute', imputer),
    ('scale', scaler),
    ('clf', clf_logreg),
])
pipe_logreg.fit(X_train, y_train)

print("Training Random Forest...")
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
pipe_rf = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('clf', clf_rf),
])
pipe_rf.fit(X_train, y_train)

# --- Evaluate Robustness ---
robustness_results = []
for cond in test_df['condition'].unique():
    mask = test_df['condition'] == cond
    X_c, y_c = X_test[mask], y_test[mask]
    
    pred_lr = pipe_logreg.predict(X_c)
    pred_rf = pipe_rf.predict(X_c)
    
    robustness_results.append({
        'condition': cond,
        'logreg_f1': f1_score(y_c, pred_lr, average='macro'),
        'rf_f1': f1_score(y_c, pred_rf, average='macro')
    })

print("\n--- Robustness Results ---")
rob_df = pd.DataFrame(robustness_results)
print(rob_df)

# Save Robustness to JSON for frontend
os.makedirs("frontend/public", exist_ok=True)
rob_df.to_json("frontend/public/robustness_results.json", orient="records")

# --- Export Canonical Trace for Frontend ---
# We'll take a subset of the dataset (e.g. one trial per condition) and run it through the LR model
print("\nExporting Canonical Trace...")
traces = []
for cond in df['condition'].unique():
    trial_id = df[df['condition'] == cond]['trial_id'].iloc[0]
    trial_data = df[df['trial_id'] == trial_id].copy()
    
    # Extract features for this trial
    trial_features = extract_window_features(trial_data)
    
    # We need to map back to original rows. Since we use a window, the first WINDOW_S-1 rows are lacking features.
    # We will pad the beginning with the first prediction.
    preds = pipe_logreg.predict(trial_features[feature_cols])
    probs = pipe_logreg.predict_proba(trial_features[feature_cols])
    
    # Probability of infiltration (sum of early, moderate, severe)
    # Classes are typically 0=none, 1=early, 2=moderate, 3=severe
    inf_probs = 1.0 - probs[:, 0]
    
    # Pad to match original trial length
    pad_len = WINDOW_S - 1
    padded_preds = np.pad(preds, (pad_len, 0), mode='edge')
    padded_probs = np.pad(inf_probs, (pad_len, 0), mode='edge')
    
    trial_data['probability'] = padded_probs
    trial_data['severity_class'] = padded_preds
    trial_data['alert'] = (trial_data['severity_class'] > 1).astype(int) # Alert if moderate or severe
    
    traces.append(trial_data)

canonical_trace = pd.concat(traces)
canonical_trace = canonical_trace[['time_s', 'temperature_c', 'pressure_adc', 'condition', 'probability', 'severity_class', 'alert']]
canonical_trace.rename(columns={'time_s': 'time_ms'}, inplace=True)
canonical_trace['time_ms'] = (canonical_trace['time_ms'] * 1000).astype(int)
canonical_trace.to_csv("frontend/public/replay_trace.csv", index=False)
print("Saved frontend/public/replay_trace.csv")

# --- Export C++ Header for ESP32 ---
print("\nExporting C++ Header...")
with open("firmware/sketch/model_params.h", "w") as f:
    f.write("// Auto-generated by scripts/export_edge_model.py\n")
    f.write("#ifndef MODEL_PARAMS_H\n")
    f.write("#define MODEL_PARAMS_H\n\n")
    
    f.write(f"#define WINDOW_SIZE {WINDOW_S}\n\n")
    
    # Scaler params
    f.write("const float SCALER_MEAN[] = {\n    ")
    f.write(", ".join([f"{x:.6f}" for x in scaler.mean_]))
    f.write("\n};\n\n")
    
    f.write("const float SCALER_SCALE[] = {\n    ")
    f.write(", ".join([f"{x:.6f}" for x in scaler.scale_]))
    f.write("\n};\n\n")
    
    # Imputer params (for NaN replacement)
    f.write("const float IMPUTER_MEAN[] = {\n    ")
    f.write(", ".join([f"{x:.6f}" for x in imputer.statistics_]))
    f.write("\n};\n\n")
    
    # LogReg params
    coef = clf_logreg.coef_
    intercept = clf_logreg.intercept_
    num_classes = coef.shape[0]
    num_features = coef.shape[1]
    
    f.write(f"const int NUM_CLASSES = {num_classes};\n")
    f.write(f"const int NUM_FEATURES = {num_features};\n\n")
    
    f.write("const float LOGREG_INTERCEPT[] = {\n    ")
    f.write(", ".join([f"{x:.6f}" for x in intercept]))
    f.write("\n};\n\n")
    
    f.write("const float LOGREG_COEF[][NUM_FEATURES] = {\n")
    for row in coef:
        f.write("    {")
        f.write(", ".join([f"{x:.6f}" for x in row]))
        f.write("},\n")
    f.write("};\n\n")
    
    f.write("#endif // MODEL_PARAMS_H\n")

print("Saved firmware/sketch/model_params.h")

# Also update the notebook (simple nbformat injection)
print("\nUpdating iv_infil_v3.ipynb with evaluation results...")
import nbformat
with open('iv_infil_v3.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

new_cell = nbformat.v4.new_markdown_cell(f"## Robustness Results (Auto-Updated)\n```\n{rob_df.to_string()}\n```")
nb.cells.append(new_cell)

with open('iv_infil_v3.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Done!")
