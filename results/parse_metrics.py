import json
from pathlib import Path

path = Path("results/workstream_b/workstream_b_metrics.json")
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== MAIN METRICS ===")
for mname, m in d["main"].items():
    dd = m.get("detection_delay", {})
    fpr = m.get("confounder_fpr", {})
    f1 = m.get("macro_f1_mean", m.get("macro_f1"))
    f1_std = m.get("macro_f1_std", 0.0)
    ba = m.get("balanced_acc_mean", m.get("balanced_acc"))
    raw_delay = m.get("delay_raw_mean_s", dd.get("mean_delay_raw_s"))
    raw_delay_std = m.get("delay_raw_std_s", 0.0)
    clamped_delay = dd.get("mean_delay_clamped_s")
    print(f"[{mname}]")
    print(f"  Macro-F1: {f1} +/- {f1_std}")
    print(f"  BalAcc: {ba}")
    print(f"  Delay Raw: {raw_delay}s +/- {raw_delay_std}s (clamped: {clamped_delay}s)")
    print(f"  Detection counts: detected={dd.get('n_detected')}, missed={dd.get('n_missed')}, premature={dd.get('n_premature_alarms')}")
    print(f"  Confounder FPR: {fpr.get('fpr')} (FP: {fpr.get('n_false_alarms')} / {fpr.get('n_confounder_windows')}), CI: [{fpr.get('ci_lower')}, {fpr.get('ci_upper')}]")
    print(f"  Confusion Matrix: {m.get('confusion_matrix')}")

print("\n=== SENSOR ABLATION ===")
print(json.dumps(d.get("sensor_ablation", {}), indent=2))

print("\n=== WINDOW ABLATION ===")
print(json.dumps(d.get("window_ablation", {}), indent=2))

print("\n=== SHUFFLE ABLATION ===")
print(json.dumps(d.get("shuffle_ablation", {}), indent=2))

print("\n=== SCARCITY 20% ===")
print(json.dumps(d.get("scarcity_20pct", {}), indent=2))

print("\n=== SCENARIO BREAKDOWN ===")
print(json.dumps(d.get("scenario_breakdown", {}), indent=2))
